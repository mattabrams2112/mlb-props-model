"""
Game View — both lineups with HRR ratings, pinch hitter detection,
bullpen stats, sportsbook line inputs, team logos, and parallel fetching.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from eastern_time import today_et, today_str_et, now_et
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False
import statsapi
import requests as _req

from feature_engineering import build_features, get_feature_cols, TARGET_COL
from tracker import add_predictions as tracker_add
from full_tracker import log_play
from pitcher_data import get_pitcher_season_stats, get_pitcher_name
from statcast_features import get_batter_statcast, get_pitcher_statcast
from weather import get_park_factor
from rating import compute_rating
from lineup_fetcher import get_todays_lineups
from team_logos import get_logo, logo_img_tag
from bvp_stats import get_bvp
from stadium_weather import get_stadium_weather
from bullpen_data import get_bullpen_stats
from ratings_cache import get_cached_rating, save_rating, clear_ratings_for_players
from odds_api import get_todays_event_ids, get_player_line, fair_probability, american_to_prob, prob_to_american, ODDS_API_KEY
from team_stats import get_team_recent_scoring, get_team_defense_rating
from umpire_data import get_game_umpire
from pitcher_data import get_pitcher_throws, get_pitcher_last_n_starts, get_pitcher_rest_days

st.set_page_config(page_title="Game View | MLB Props", page_icon="🎯", layout="wide")
st.markdown("""
<style>
  .block-container{padding-top:1rem;}
  h1,h2,h3,h4{color:#38bdf8!important;}
  .stMarkdown p,label,.stCaption{color:#7dd3fc!important;}
  .game-header{background:#1e293b;border:1px solid #1e40af;border-radius:10px;
               padding:12px 18px;margin-bottom:4px;}
</style>""", unsafe_allow_html=True)

MLB_API = 'https://statsapi.mlb.com/api/v1'


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=900)
def _get_lineups(date_str: str):
    """Cache lineup data for 15 min — survives browser close as long as Railway is running."""
    return get_todays_lineups(date_str)


@st.cache_data(show_spinner=False, ttl=86400)
def get_player_info(pid: int) -> tuple:
    try:
        info = statsapi.lookup_player(pid)
        if info:
            return info[0]['fullName'], info[0].get('currentTeam', {}).get('abbreviation', '')
    except Exception:
        pass
    return str(pid), ''


@st.cache_data(show_spinner=False, ttl=7200)
def fetch_logs(player_id: int) -> pd.DataFrame:
    current_year = datetime.now().year
    seasons = [current_year - 2, current_year - 1, current_year]
    rows = []
    for season in seasons:
        try:
            resp = _req.get(f'{MLB_API}/people/{player_id}/stats',
                            params={'stats': 'gameLog', 'group': 'hitting', 'season': season},
                            timeout=15)
            resp.raise_for_status()
            splits = (resp.json().get('stats') or [{}])[0].get('splits', [])
            for s in splits:
                stat = s.get('stat', {}); gi = s.get('game', {})
                ih = s.get('isHome', True)
                pt = s.get('team', {}).get('abbreviation', '')
                op = s.get('opponent', {}).get('abbreviation', '')
                rows.append({
                    'player_id': player_id, 'season': season,
                    'date':      gi.get('gameDate', s.get('date', '')),
                    'game_pk':   str(gi.get('gamePk', '')),
                    'opponent':  op, 'home_team': pt if ih else op,
                    'is_home':   int(ih),
                    'ab':  int(stat.get('atBats', 0)),
                    'h':   int(stat.get('hits', 0)),
                    'r':   int(stat.get('runs', 0)),
                    'rbi': int(stat.get('rbi', 0)),
                    'd':   int(stat.get('doubles', 0)),
                    't':   int(stat.get('triples', 0)),
                    'hr':  int(stat.get('homeRuns', 0)),
                    'bb':  int(stat.get('baseOnBalls', 0)),
                    'k':   int(stat.get('strikeOuts', 0)),
                    'sb':  int(stat.get('stolenBases', 0)),
                })
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df[df['ab'] > 0].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def run_prediction(player_id: int, pitcher_id, is_home: bool, park_team: str,
                   temp_f: float, wind_speed: float, wind_dir: int,
                   game_date: str = ''):
    df = fetch_logs(player_id)
    if df.empty or len(df) < 25:
        return None

    # Freeze ratings at pre-game state — exclude game day and later
    if game_date:
        try:
            from datetime import date as date_type
            cutoff = pd.Timestamp(game_date).date()
            df = df[df['date'].dt.date < cutoff].copy()
        except Exception:
            pass

    if len(df) < 25:
        return None

    df_feat = build_features(df, fetch_weather=False,
                              override_pitcher_id=pitcher_id, fast_mode=True)
    idx = df_feat.index[-1]
    df_feat.at[idx, 'is_home']     = int(is_home)
    df_feat.at[idx, 'park_factor'] = get_park_factor(park_team)
    df_feat.at[idx, 'temp_f']      = temp_f
    df_feat.at[idx, 'wind_speed']  = wind_speed
    df_feat.at[idx, 'wind_dir']    = wind_dir

    fc = get_feature_cols()
    dc = df_feat.dropna(subset=fc).reset_index(drop=True)
    if len(dc) < 20:
        return None

    X = dc[fc].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = dc[TARGET_COL]

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X, y)

    if HAS_LGBM:
        lgb = LGBMRegressor(n_estimators=100, learning_rate=0.08, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbose=-1)
        lgb.fit(X, y)

    latest   = dc.iloc[-1:].copy()
    latest.at[latest.index[0], 'is_home']     = int(is_home)
    latest.at[latest.index[0], 'park_factor'] = get_park_factor(park_team)
    latest.at[latest.index[0], 'temp_f']      = temp_f
    latest.at[latest.index[0], 'wind_speed']  = wind_speed
    latest.at[latest.index[0], 'wind_dir']    = wind_dir

    latest_X  = latest[fc].apply(pd.to_numeric, errors='coerce').fillna(0)
    xgb_pred  = float(xgb.predict(latest_X)[0])
    if HAS_LGBM:
        lgb_pred = float(lgb.predict(latest_X)[0])
        proj = max(0.0, (xgb_pred * 0.55 + lgb_pred * 0.45))  # slight XGBoost bias
    else:
        proj = max(0.0, xgb_pred)

    # Projection floor — can't be less than 30% of season avg or 30% of 30g avg
    # Prevents bad feature values from producing absurdly low projections
    season_avg_val = float(dc['total_season_avg'].iloc[-1]) if not np.isnan(dc['total_season_avg'].iloc[-1]) else 0
    r30_avg = float((df.tail(30)['h'] + df.tail(30)['r'] + df.tail(30)['rbi']).mean())

    # Floor — can't go below 30% of recent averages
    floor = max(season_avg_val * 0.30, r30_avg * 0.30)
    proj  = max(proj, floor)

    # Ceiling — can't exceed 1.5x the 30g avg or 3.5 absolute max
    # 3.5 is a very strong game (e.g. 2H + 1R + 1RBI) — realistic upper bound
    ceiling = min(3.5, max(r30_avg * 1.5, season_avg_val * 1.5, 1.5))
    proj    = min(proj, ceiling)

    r7  = df.tail(7);  hrr7  = (r7['h']  + r7['r']  + r7['rbi']).mean()
    r30 = df.tail(30); hrr30 = (r30['h'] + r30['r'] + r30['rbi']).mean()
    ab30 = r30['ab'].sum(); h30 = r30['h'].sum()

    # Home/away splits
    home_games = df[df['is_home'] == 1]
    away_games = df[df['is_home'] == 0]
    home_hrr = (home_games['h'] + home_games['r'] + home_games['rbi']).mean() if len(home_games) >= 5 else None
    away_hrr = (away_games['h'] + away_games['r'] + away_games['rbi']).mean() if len(away_games) >= 5 else None

    return {
        'proj':     round(proj, 2),
        'r7g':      round(float(hrr7), 2),
        'r30g':     round(float(hrr30), 2),
        'savg':     round(float(dc['total_season_avg'].iloc[-1]), 2)
                    if not np.isnan(dc['total_season_avg'].iloc[-1]) else 0.0,
        'ba30':     round(h30 / ab30, 3) if ab30 > 0 else 0.250,
        'home_hrr': round(float(home_hrr), 2) if home_hrr else None,
        'away_hrr': round(float(away_hrr), 2) if away_hrr else None,
        'df':       df,
        'r20g_venue': round(float(dc['hrr_20g_venue'].iloc[-1]), 2)
                      if 'hrr_20g_venue' in dc.columns and not np.isnan(dc['hrr_20g_venue'].iloc[-1]) else None,
        'ba_venue':   round(float(dc['ba_20g_venue'].iloc[-1]), 3)
                      if 'ba_20g_venue' in dc.columns and not np.isnan(dc['ba_20g_venue'].iloc[-1]) else None,
    }


def get_rating(res, player_id, pitcher_id, park_team, batting_order,
               temp_f, wind_speed, wind_dir, bp_era=4.20, bp_whip=1.30,
               line=None, over_odds=None, is_home=True,
               opp_fip=4.20, opp_last3_era=4.30, opp_last3_whip=1.28,
               pitcher_throws='R',
               batter_xba_vs_rhp=0.250, batter_xba_vs_lhp=0.250,
               batter_hard_hit_vs_rhp=0.360, batter_hard_hit_vs_lhp=0.360,
               team_runs_avg=4.5, umpire_tendency=0.0,
               opp_def_rating=0.0, pitcher_rest_factor=0.0,
               pitcher_gb_pct=0.430):
    season = int(res['df']['season'].iloc[-1])
    b_sc   = get_batter_statcast(player_id, season)
    p_sc   = get_pitcher_statcast(pitcher_id, season) if pitcher_id else {}
    p_std  = get_pitcher_season_stats(pitcher_id, season) if pitcher_id else {}
    bvp    = get_bvp(player_id, pitcher_id) if pitcher_id else {}
    return compute_rating(
        recent_7g         = res['r7g'],
        recent_30g        = res['r30g'],
        season_avg        = res['savg'],
        opp_era           = p_std.get('opp_era', 4.30),
        opp_whip          = p_std.get('opp_whip', 1.28),
        batter_fb_barrel  = b_sc.get('batter_fb_barrel_pct', 0.080),
        batter_bk_barrel  = b_sc.get('batter_bk_barrel_pct', 0.040),
        batter_os_barrel  = b_sc.get('batter_os_barrel_pct', 0.050),
        pitcher_fb_barrel = p_sc.get('pitcher_fb_barrel_pct', 0.080),
        pitcher_bk_barrel = p_sc.get('pitcher_bk_barrel_pct', 0.040),
        pitcher_os_barrel = p_sc.get('pitcher_os_barrel_pct', 0.050),
        batter_fb_seen    = b_sc.get('batter_fb_seen_pct', 0.55),
        batter_bk_seen    = b_sc.get('batter_bk_seen_pct', 0.25),
        batter_os_seen    = b_sc.get('batter_os_seen_pct', 0.20),
        park_factor       = get_park_factor(park_team),
        wind_speed        = wind_speed,
        wind_dir          = wind_dir,
        bvp_avg              = bvp.get('bvp_avg', 0.250),
        bvp_sample           = bvp.get('bvp_sample', 0),
        batting_order        = batting_order,
        recent_ba            = res['ba30'],
        temp_f               = temp_f,
        projection           = res['proj'],
        bp_era               = bp_era,
        bp_whip              = bp_whip,
        line                 = line,
        over_odds            = over_odds,
        home_hrr             = res.get('home_hrr'),
        away_hrr             = res.get('away_hrr'),
        is_home              = is_home,
        recent_20g           = res.get('r20g_venue'),
        recent_ba_venue      = res.get('ba_venue'),
        batter_hard_hit_pct  = b_sc.get('batter_hard_hit_pct', 0.360),
        pitcher_hard_hit_pct = p_sc.get('pitcher_hard_hit_pct', 0.360),
        batter_xba           = b_sc.get('batter_xba', 0.250),
        pitcher_xba_allowed  = p_sc.get('pitcher_xba_allowed', 0.250),
        batter_avg_ev           = b_sc.get('batter_avg_ev', 88.0),
        pitcher_avg_ev          = p_sc.get('pitcher_avg_ev', 88.0),
        opp_fip                 = opp_fip,
        opp_last3_era           = opp_last3_era,
        opp_last3_whip          = opp_last3_whip,
        pitcher_throws          = pitcher_throws,
        batter_xba_vs_rhp       = batter_xba_vs_rhp,
        batter_xba_vs_lhp       = batter_xba_vs_lhp,
        batter_hard_hit_vs_rhp  = batter_hard_hit_vs_rhp,
        batter_hard_hit_vs_lhp  = batter_hard_hit_vs_lhp,
        team_runs_avg           = team_runs_avg,
        umpire_tendency         = umpire_tendency,
        opp_def_rating          = opp_def_rating,
        pitcher_rest_factor     = pitcher_rest_factor,
        pitcher_gb_pct          = pitcher_gb_pct,
    )


def cv(v, high, med):
    return '#22c55e' if v >= high else '#eab308' if v >= med else '#ef4444'


def render_lineup(container, batter_ids, batter_codes, is_home, opp_pitcher_id,
                  opp_team, park_team, weather, game_label, opp_p_name,
                  date_key: str, batter_team: str = '', game_date: str = '',
                  event_id: str = '', game_pk: str = ''):

    season      = datetime.now().year
    p_std       = get_pitcher_season_stats(opp_pitcher_id) if opp_pitcher_id else {}
    p_sc        = get_pitcher_statcast(opp_pitcher_id) if opp_pitcher_id else {}
    bp          = get_bullpen_stats(opp_team, season)
    bp_era      = bp.get('bp_era', 4.20)
    bp_whip     = bp.get('bp_whip', 1.30)
    p_throws    = get_pitcher_throws(opp_pitcher_id) if opp_pitcher_id else 'R'
    p_last3     = get_pitcher_last_n_starts(opp_pitcher_id, 3, season) if opp_pitcher_id else {}
    p_rest      = get_pitcher_rest_days(opp_pitcher_id, season, game_date) if opp_pitcher_id else {}
    team_score  = get_team_recent_scoring(batter_team)
    opp_defense = get_team_defense_rating(opp_team, season)
    try:
        ump_data = get_game_umpire(int(game_pk)) if game_pk else {}
    except Exception:
        ump_data = {}

    era   = f"{p_std.get('opp_era',0):.2f}"   if opp_pitcher_id else '—'
    whip  = f"{p_std.get('opp_whip',0):.2f}"  if opp_pitcher_id else '—'
    kpct  = f"{p_std.get('opp_k_pct',0):.1%}" if opp_pitcher_id else '—'
    fb_t  = p_sc.get('pitcher_fb_thrown_pct', 0)
    bk_t  = p_sc.get('pitcher_bk_thrown_pct', 0)
    os_t  = p_sc.get('pitcher_os_thrown_pct', 0)

    pitcher_bar = (
        f'<div style="background:#1e293b;border-radius:6px;padding:8px 12px;'
        f'margin-bottom:4px;font-size:12px;color:#7dd3fc;">'
        f'⚾ <b style="color:#38bdf8;">{opp_p_name}</b>'
        f' &nbsp;ERA {era} &nbsp;WHIP {whip} &nbsp;K% {kpct}'
        f'<span style="margin-left:14px;color:#94a3b8;">'
        f'Pitch mix → FB {fb_t:.0%} BK {bk_t:.0%} OS {os_t:.0%}</span>'
        f'</div>'
        f'<div style="background:#0f172a;border-radius:6px;padding:5px 12px;'
        f'margin-bottom:6px;font-size:11px;color:#94a3b8;">'
        f'Bullpen ERA {bp_era:.2f} · WHIP {bp_whip:.2f} · '
        f'<span style="color:{"#22c55e" if bp_era>=4.5 else "#eab308" if bp_era>=3.8 else "#ef4444"};">'
        f'{"🔥 Hitter-friendly bullpen" if bp_era>=4.5 else "⚖ Average bullpen" if bp_era>=3.8 else "🔒 Tough bullpen"}'
        f'</span></div>'
    )

    header = (
        '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
        '<tr style="background:#0f172a;color:#38bdf8;font-size:11px;font-weight:700;">'
    )
    for h, align in [('#','c'),('','l'),('Player','l'),('Ord','c'),
                     ('Rating','c'),('Proj','c'),('Line','c'),('Odds','c'),
                     ('Fair Odds','c'),('Edge','c'),
                     ('30g BA','c'),('7g HRR','c'),('30g HRR','c'),('Barrel % (seen)','l')]:
        a = 'left' if align == 'l' else 'center'
        header += f'<th style="padding:5px 7px;text-align:{a};border-bottom:1px solid #1e40af;">{h}</th>'
    header += '</tr>'

    placeholder = container.empty()
    placeholder.markdown(
        pitcher_bar + header +
        '<tr><td colspan="12" style="padding:12px;color:#475569;text-align:center;">'
        '⏳ Fetching all batters in parallel...</td></tr></table>',
        unsafe_allow_html=True
    )

    # ── Parallel fetch (cached in session state for instant return visits) ────
    game_started    = status in ('In Progress', 'Manager Challenge', 'Final',
                                 'Game Over', 'Completed Early')
    fetch_cache_key = f'gv_fetch_{date_key}_{game_pk}_{int(is_home)}'

    if fetch_cache_key in st.session_state:
        fetched = st.session_state[fetch_cache_key]
    else:
        def fetch(args):
            idx, pid = args
            pname, pteam = get_player_info(pid)
            ocode      = batter_codes.get(int(pid), (idx + 1) * 100)
            is_starter = (ocode % 100 == 0)
            spot       = ocode // 100
            sub_idx    = ocode % 100
            res        = run_prediction(pid, opp_pitcher_id, is_home, park_team,
                                        weather['temp_f'], weather['wind_speed'],
                                        weather['wind_dir_code'], game_date=game_date)
            odds_data = (get_player_line(pname, event_id)
                         if ODDS_API_KEY and event_id and not game_started else None)
            return idx, pid, pname, pteam, res, is_starter, spot, sub_idx, odds_data

        with ThreadPoolExecutor(max_workers=2) as exe:
            fetched = list(exe.map(fetch, enumerate(batter_ids)))
        st.session_state[fetch_cache_key] = fetched

    # Sort: starters by spot, subs by spot then sub_idx
    fetched.sort(key=lambda x: (0 if x[5] else 1, x[6], x[7]))

    # ── Lineup context ────────────────────────────────────────────────────────
    # Build team projection map: spot -> proj for all starters with results
    _LEAGUE_AVG    = 1.8
    _starter_projs = {sp: r['proj'] for _, _, _, _, r, is_s, sp, _, _ in fetched
                      if is_s and r and r.get('proj') is not None}
    _team_avg      = (sum(_starter_projs.values()) / len(_starter_projs)
                      if _starter_projs else _LEAGUE_AVG)

    # ── Build rows ────────────────────────────────────────────────────────────
    rows_html = []
    totals    = []

    for row_i, (idx, pid, pname, pteam, res, is_starter, spot, sub_idx, odds_data) in enumerate(fetched):
        display_order = str(spot) if is_starter else f'{spot}.{sub_idx}'
        batting_order = spot if is_starter else 0

        # Lineup context — how strong is this player's surrounding lineup?
        if is_starter and res and spot in _starter_projs:
            _prev = 9 if spot == 1 else spot - 1
            _next = 1 if spot == 9 else spot + 1
            _nbrs = [_starter_projs[s] for s in (_prev, _next) if s in _starter_projs]
            _nbr_avg  = sum(_nbrs) / len(_nbrs) if _nbrs else _team_avg
            _ctx_avg  = 0.5 * _team_avg + 0.5 * _nbr_avg
            _ctx_pct  = max(-0.12, min(0.12, (_ctx_avg - _LEAGUE_AVG) / _LEAGUE_AVG * 0.4))
        else:
            _ctx_pct  = 0.0
        _disp_proj = res['proj'] if res else 0  # updated in each branch below

        bg = '#0f172a' if row_i % 2 == 0 else '#1e293b'
        if not is_starter:
            bg = '#111827'

        logo        = logo_img_tag(batter_team or pteam, 24)
        order_color = ('#22c55e' if spot <= 2 and is_starter else
                       '#38bdf8' if spot <= 5 and is_starter else '#475569')

        line_key = f'line_{date_key}_{pid}'
        line_val = st.session_state.get(line_key)

        if is_starter and res:
            session_key = f'locked_{date_key}_{pid}'

            # Try cache sources in order
            cached = (st.session_state.get(session_key) or
                      (get_cached_rating(game_date, pid) if game_date else None))

            if cached:
                # Always use locked pre-game rating — never recalculate
                locked_rating, locked_grade, locked_proj = cached
                r_data = {'total': locked_rating, 'grade': locked_grade,
                          'color': '#22c55e' if locked_rating >= 75 else '#eab308' if locked_rating >= 55 else '#ef4444',
                          'components': {}, 'line_label': None}
                res = dict(res); res['proj'] = locked_proj
                _disp_proj = locked_proj
            elif game_started:
                # Game started, no pre-game cache — calculate without odds
                book_line  = None
                book_odds  = None
                season_r   = int(res['df']['season'].iloc[-1])
                b_sc_local = get_batter_statcast(pid, season_r)
                _res_ctx   = dict(res)
                _res_ctx['proj'] = round(max(0.5, res['proj'] * (1 + _ctx_pct)), 2)
                r_data = get_rating(_res_ctx, pid, opp_pitcher_id, park_team, batting_order,
                                    weather['temp_f'], weather['wind_speed'],
                                    weather['wind_dir_code'],
                                    bp_era=bp_era, bp_whip=bp_whip,
                                    line=None, over_odds=None,
                                    is_home=is_home,
                                    opp_fip=p_std.get('opp_fip', 4.20),
                                    opp_last3_era=p_last3.get('opp_last3_era', 4.30),
                                    opp_last3_whip=p_last3.get('opp_last3_whip', 1.28),
                                    pitcher_throws=p_throws,
                                    batter_xba_vs_rhp=b_sc_local.get('batter_xba_vs_rhp', 0.250),
                                    batter_xba_vs_lhp=b_sc_local.get('batter_xba_vs_lhp', 0.250),
                                    batter_hard_hit_vs_rhp=b_sc_local.get('batter_hard_hit_vs_rhp', 0.360),
                                    batter_hard_hit_vs_lhp=b_sc_local.get('batter_hard_hit_vs_lhp', 0.360),
                                    team_runs_avg=team_score.get('team_runs_avg', 4.5),
                                    umpire_tendency=ump_data.get('umpire_tendency', 0.0),
                                    opp_def_rating=opp_defense.get('def_rating', 0.0),
                                    pitcher_rest_factor=p_rest.get('rest_factor', 0.0),
                                    pitcher_gb_pct=p_sc.get('pitcher_gb_pct', 0.430))
                _disp_proj = _res_ctx['proj']
                st.session_state[session_key] = (r_data['total'], r_data['grade'], _disp_proj)
                if game_date and opp_p_name != 'TBD':
                    save_rating(game_date, pid, r_data['total'], r_data['grade'],
                                _disp_proj, player_name=pname, team=batter_team,
                                vs_pitcher=opp_p_name)
            else:
                book_line  = odds_data['line']      if odds_data else line_val
                book_odds  = odds_data['over_odds'] if odds_data else None
                season_r   = int(res['df']['season'].iloc[-1])
                b_sc_local = get_batter_statcast(pid, season_r)
                _res_ctx   = dict(res)
                _res_ctx['proj'] = round(max(0.5, res['proj'] * (1 + _ctx_pct)), 2)
                r_data = get_rating(_res_ctx, pid, opp_pitcher_id, park_team, batting_order,
                                    weather['temp_f'], weather['wind_speed'],
                                    weather['wind_dir_code'],
                                    bp_era=bp_era, bp_whip=bp_whip,
                                    line=book_line, over_odds=book_odds,
                                    is_home=is_home,
                                    opp_fip=p_std.get('opp_fip', 4.20),
                                    opp_last3_era=p_last3.get('opp_last3_era', 4.30),
                                    opp_last3_whip=p_last3.get('opp_last3_whip', 1.28),
                                    pitcher_throws=p_throws,
                                    batter_xba_vs_rhp=b_sc_local.get('batter_xba_vs_rhp', 0.250),
                                    batter_xba_vs_lhp=b_sc_local.get('batter_xba_vs_lhp', 0.250),
                                    batter_hard_hit_vs_rhp=b_sc_local.get('batter_hard_hit_vs_rhp', 0.360),
                                    batter_hard_hit_vs_lhp=b_sc_local.get('batter_hard_hit_vs_lhp', 0.360),
                                    team_runs_avg=team_score.get('team_runs_avg', 4.5),
                                    umpire_tendency=ump_data.get('umpire_tendency', 0.0),
                                    opp_def_rating=opp_defense.get('def_rating', 0.0),
                                    pitcher_rest_factor=p_rest.get('rest_factor', 0.0),
                                    pitcher_gb_pct=p_sc.get('pitcher_gb_pct', 0.430))
                _disp_proj = _res_ctx['proj']
                # Lock in session state immediately
                st.session_state[session_key] = (r_data['total'], r_data['grade'], _disp_proj)
                # Only freeze rating once pitcher is confirmed — TBD ratings may change
                if game_date and opp_p_name != 'TBD':
                    save_rating(game_date, pid, r_data['total'], r_data['grade'],
                                _disp_proj, player_name=pname, team=batter_team,
                                vs_pitcher=opp_p_name)

            # Log ALL plays to analytics tracker — only when pitcher is confirmed
            if pname and game_date and opp_p_name != 'TBD':
                _pre_game = status in ('Preview', 'Pre-Game', 'Scheduled', 'Warmup')
                try:
                    log_play(
                        player=pname, team=batter_team,
                        rating=r_data['total'], grade=r_data['grade'],
                        projected=_disp_proj,
                        line=disp_line, over_odds=disp_odds,
                        vs_pitcher=opp_p_name, is_home=is_home,
                        game_date=game_date,
                        game_started=not _pre_game,
                    )
                except Exception:
                    pass

            # Add qualifying players to the tracker if game is completed or in the past
            _game_finished = status in ('Final', 'Game Over', 'Completed Early')
            from datetime import datetime as _dt
            _today = today_str_et()
            _r = r_data['total']; _p = _disp_proj
            _qualifies = ((70 <= _r <= 74 and _p >= 3.0) or
                          (80 <= _r <= 84 and _p >= 1.5) or
                          (85 <= _r <= 89 and _p >= 1.5))
            if _qualifies:
                _units    = 2.0 if 85 <= _r <= 89 else 1.0
                _bet      = int(_units * 8)
                _u_str    = '1.5' if _units == 1.5 else str(int(_units))
                _stake_badge = (f' <span style="font-size:10px;background:#1e3a5f;color:#7dd3fc;'
                                f'border-radius:3px;padding:1px 4px;font-weight:700;">'
                                f'{_u_str}u · ${_bet}</span>')
            if _qualifies and pname and game_date and (game_date < _today or _game_finished):
                try:
                    tracker_add([{
                        'player':     pname,
                        'team':       batter_team,
                        'rating':     r_data['total'],
                        'grade':      r_data['grade'],
                        'projected':  _disp_proj,
                        'vs_pitcher': opp_p_name,
                        'line':       disp_line,
                        'over_odds':  disp_odds,
                    }], game_date=game_date)
                except Exception:
                    pass

            # Always sync tracker with the freshly recalculated rating.
            # Runs whenever rating was computed (not from cache) so recalculates
            # update the tracker — including dropping players that no longer qualify.
            if not cached and pname and game_date and opp_p_name != 'TBD':
                try:
                    from tracker import update_rating_if_exists as _tracker_sync
                    _tracker_sync(pname, game_date, r_data['total'], r_data['grade'],
                                  _disp_proj, opp_p_name)
                except Exception:
                    pass

            batter_sc = get_batter_statcast(pid, int(res['df']['season'].iloc[-1]))
            fb_b = batter_sc.get('batter_fb_barrel_pct', 0)
            bk_b = batter_sc.get('batter_bk_barrel_pct', 0)
            os_b = batter_sc.get('batter_os_barrel_pct', 0)
            fb_s = batter_sc.get('batter_fb_seen_pct', 0)
            bk_s = batter_sc.get('batter_bk_seen_pct', 0)
            os_s = batter_sc.get('batter_os_seen_pct', 0)

            rc = cv(r_data['total'], 75, 55)
            pc = cv(_disp_proj, 3.0, 2.0)
            bc = cv(res['ba30'], 0.280, 0.250)

            # Line / odds display
            disp_line = odds_data['line'] if odds_data else line_val
            disp_odds = odds_data['over_odds'] if odds_data else None

            if disp_line is not None:
                edge      = round(res['proj'] - disp_line, 2)
                edge_str  = f'{edge:+.2f}'
                ec        = '#22c55e' if edge > 0.25 else '#eab308' if edge > 0 else '#ef4444'
                line_display = f'<span style="color:#e0f2fe;">{disp_line}</span>'
                edge_display = f'<span style="color:{ec};font-weight:700;">{edge_str}</span>'
            else:
                line_display = '<span style="color:#475569;">—</span>'
                edge_display = '<span style="color:#475569;">—</span>'

            if disp_odds is not None:
                odds_color   = '#22c55e' if disp_odds > 0 else '#7dd3fc'
                odds_display = f'<span style="color:{odds_color};font-weight:700;">{disp_odds:+d}</span>'
                fair_p       = fair_probability(res['proj'], disp_line) if disp_line else 0
                fair_o       = prob_to_american(fair_p)
                fair_color   = '#22c55e' if fair_p > american_to_prob(disp_odds) else '#94a3b8'
                fair_display = f'<span style="color:{fair_color};">{fair_o:+d}</span>'
            else:
                odds_display = '<span style="color:#475569;">—</span>'
                fair_display = '<span style="color:#475569;">—</span>'

            barrel_html = (
                f'<div style="font-size:10px;line-height:1.7;">'
                f'FB {fb_b:.1%} <span style="color:#475569;">({fb_s:.0%})</span><br>'
                f'BK {bk_b:.1%} <span style="color:#475569;">({bk_s:.0%})</span><br>'
                f'OS {os_b:.1%} <span style="color:#475569;">({os_s:.0%})</span>'
                f'</div>'
            )

            _name_cell_style = (
                'padding:6px 8px;font-weight:700;white-space:nowrap;color:#fbbf24;'
                if _qualifies else
                'padding:6px 8px;color:#e0f2fe;font-weight:600;white-space:nowrap;'
            )
            _name_badge = (f' <span style="font-size:10px;background:#f59e0b;color:#000;border-radius:3px;padding:1px 4px;font-weight:800;">BET</span>{_stake_badge}'
                           if _qualifies else '')
            _row_border = 'border-left:3px solid #f59e0b;' if _qualifies else ''
            row = (
                f'<tr style="background:{bg};border-bottom:1px solid #1e293b;{_row_border}">'
                f'<td style="padding:6px 8px;color:#475569;font-size:12px;">{display_order}</td>'
                f'<td style="padding:6px 8px;">{logo}</td>'
                f'<td style="{_name_cell_style}">'
                f'{pname}{_name_badge}<div style="font-size:10px;color:#475569;">{game_label}</div></td>'
                f'<td style="padding:6px 8px;text-align:center;color:{order_color};font-weight:700;">#{batting_order}</td>'
                f'<td style="padding:6px 8px;text-align:center;font-weight:800;color:{rc};">'
                f'{r_data["total"]} <span style="font-size:10px;">{r_data["grade"]}</span></td>'
                f'<td style="padding:6px 8px;text-align:center;font-weight:800;font-size:15px;color:{pc};">{_disp_proj}</td>'
                f'<td style="padding:6px 8px;text-align:center;">{line_display}</td>'
                f'<td style="padding:6px 8px;text-align:center;">{odds_display}</td>'
                f'<td style="padding:6px 8px;text-align:center;">{fair_display}</td>'
                f'<td style="padding:6px 8px;text-align:center;">{edge_display}</td>'
                f'<td style="padding:6px 8px;text-align:center;color:{bc};font-size:12px;">.{int(res["ba30"]*1000):03d}</td>'
                f'<td style="padding:6px 8px;text-align:center;color:#7dd3fc;font-size:12px;">{res["r7g"]}</td>'
                f'<td style="padding:6px 8px;text-align:center;color:#7dd3fc;font-size:12px;">{res["r30g"]}</td>'
                f'<td style="padding:6px 8px;">{barrel_html}</td>'
                f'</tr>'
            )
            totals.append((r_data['total'], _disp_proj))

        elif not is_starter:
            row = (
                f'<tr style="background:{bg};opacity:0.6;">'
                f'<td style="padding:4px 8px;color:#475569;font-size:11px;">{display_order}</td>'
                f'<td style="padding:4px 8px;">{logo}</td>'
                f'<td style="color:#94a3b8;padding:4px 8px;font-size:12px;">{pname}'
                f'<div style="font-size:10px;color:#475569;">PH for spot {spot}</div></td>'
                f'<td colspan="9" style="padding:4px 8px;color:#475569;font-size:11px;font-style:italic;">'
                f'Pinch hitter — excluded from totals</td>'
                f'</tr>'
            )
        else:
            row = (
                f'<tr style="background:{bg};">'
                f'<td style="padding:6px 8px;color:#475569;">{display_order}</td>'
                f'<td style="padding:6px 8px;">{logo}</td>'
                f'<td style="color:#e0f2fe;padding:6px 8px;">{pname}</td>'
                f'<td colspan="9" style="padding:6px 8px;color:#475569;font-size:11px;">Not enough data</td>'
                f'</tr>'
            )
        rows_html.append(row)

    totals_row = ''
    if totals:
        avg_r = round(sum(t[0] for t in totals) / len(totals))
        tot_p = round(sum(t[1] for t in totals), 2)
        rc    = cv(avg_r, 75, 55); pc = cv(tot_p / max(len(totals), 1), 3.0, 2.0)
        # Save team HRR total to session state for Game Predictions page
        st.session_state[f'team_hrr_{date_key}_{batter_team}'] = tot_p
        totals_row = (
            f'<tr style="background:#0f172a;border-top:2px solid #1e40af;">'
            f'<td colspan="3" style="padding:7px;color:#38bdf8;font-weight:700;">LINEUP TOTALS</td>'
            f'<td></td>'
            f'<td style="padding:7px;text-align:center;font-weight:800;color:{rc};">{avg_r} avg</td>'
            f'<td style="padding:7px;text-align:center;font-weight:800;font-size:15px;color:{pc};">{tot_p}</td>'
            f'<td colspan="6"></td></tr>'
        )

    placeholder.markdown(
        pitcher_bar + header + ''.join(rows_html) + totals_row + '</table>',
        unsafe_allow_html=True
    )

    # ── Line inputs ───────────────────────────────────────────────────────────
    starters_with_data = [(idx, pid, pname, res) for idx, pid, pname, _, res, is_s, _, _, _
                          in fetched if is_s and res]
    if starters_with_data:
        with st.expander('📥 Enter Sportsbook Lines', expanded=False):
            st.caption('Enter the H+R+RBI line for each player. Ratings and Edge update automatically.')
            cols = st.columns(3)
            for i, (_, pid, pname, res) in enumerate(starters_with_data):
                line_key = f'line_{date_key}_{game_pk}_{batter_team}_{pid}'
                with cols[i % 3]:
                    val = st.number_input(
                        pname, min_value=0.5, max_value=6.0,
                        step=0.5, value=float(st.session_state.get(line_key, 1.5)),
                        key=f'input_{line_key}',
                    )
                    st.session_state[line_key] = val


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown('## 🎯 Game View — HRR Projections')
st.caption('Factors: hit rate · barrel rates · starter · bullpen · live weather · park · wind · batting order · sportsbook line')

hdr_col, date_col, btn_col = st.columns([3, 2, 1])
with date_col:
    selected_date = st.date_input('Date', value=today_et(),
                                  max_value=today_et(),
                                  label_visibility='collapsed')
with btn_col:
    if st.button('🔄 Refresh', use_container_width=True):
        # Clear all game view caches
        _get_lineups.clear()
        for k in list(st.session_state.keys()):
            if k.startswith('gv_'):
                st.session_state.pop(k, None)
        st.rerun()

date_str = selected_date.strftime('%m/%d/%Y')

with st.spinner(f'Fetching lineups for {selected_date.strftime("%B %d, %Y")}...'):
    games = _get_lineups(date_str)

st.markdown(f"### {selected_date.strftime('%A, %B %d, %Y')}")

if not games:
    st.warning('No games found.')
    st.stop()

has_lineups = any(g.get('home_batters') or g.get('away_batters') for g in games)
if not has_lineups:
    msg = ('No lineup data for this date.'
           if selected_date < today_et()
           else 'Lineups not yet posted. Check back 2–3 hours before first pitch.')
    st.info(msg)
    st.stop()

date_key   = selected_date.strftime('%Y%m%d')
event_map  = get_todays_event_ids() if ODDS_API_KEY else {}

for game in games:
    away     = game.get('away_team', '?')
    home     = game.get('home_team', '?')
    away_pid = game.get('away_pitcher_id')
    home_pid = game.get('home_pitcher_id')
    away_p   = get_pitcher_name(away_pid) if away_pid else 'TBD'
    home_p   = get_pitcher_name(home_pid) if home_pid else 'TBD'
    ab_ids   = game.get('away_batters', [])
    hb_ids   = game.get('home_batters', [])
    h_codes  = game.get('home_batter_codes', {})
    a_codes  = game.get('away_batter_codes', {})

    if not ab_ids and not hb_ids:
        continue

    is_past       = selected_date < datetime.now().date()
    weather       = get_stadium_weather(home, '' if is_past else game.get('start_time', ''))
    # Match event to odds API — try abbreviation, then team nickname
    TEAM_NICKNAMES = {
        'ARI':'Diamondbacks','ATL':'Braves','BAL':'Orioles','BOS':'Red Sox',
        'CHC':'Cubs','CWS':'White Sox','CIN':'Reds','CLE':'Guardians',
        'COL':'Rockies','DET':'Tigers','HOU':'Astros','KC':'Royals',
        'LAA':'Angels','LAD':'Dodgers','MIA':'Marlins','MIL':'Brewers',
        'MIN':'Twins','NYM':'Mets','NYY':'Yankees','OAK':'Athletics',
        'PHI':'Phillies','PIT':'Pirates','SD':'Padres','SEA':'Mariners',
        'SF':'Giants','STL':'Cardinals','TB':'Rays','TEX':'Rangers',
        'TOR':'Blue Jays','WSH':'Nationals',
    }
    nickname = TEAM_NICKNAMES.get(home, '')
    event_id = (event_map.get(home) or event_map.get(nickname) or
                event_map.get(home.upper()) or '')
    pf            = get_park_factor(home)
    status        = game.get('status', '')
    away_score    = game.get('away_score', '')
    home_score    = game.get('home_score', '')
    is_final      = status in ('Final', 'Game Over', 'Completed Early')

    w_txt = ('🏟️ Dome' if weather['is_dome']
             else str(weather['temp_f']) + '°F · 💨 ' + weather['wind_label']
             + ' · ' + weather['condition'])

    if is_final and away_score != '' and home_score != '':
        a_s, h_s = int(away_score), int(home_score)
        score_html = (
            f'&nbsp;&nbsp;<span style="font-size:22px;font-weight:900;'
            f'color:{"#22c55e" if a_s>h_s else "#94a3b8"};">{a_s}</span>'
            f'<span style="color:#475569;font-size:16px;margin:0 6px;">-</span>'
            f'<span style="font-size:22px;font-weight:900;'
            f'color:{"#22c55e" if h_s>a_s else "#94a3b8"};">{h_s}</span>'
        )
    else:
        score_html = ''

    status_tag = ('🏁 Final' if is_final else
                  '✅ Official' if game.get('lineups_official') else '⏳ Probable')
    status_color = '#38bdf8' if is_final else '#22c55e' if game.get('lineups_official') else '#eab308'

    st.markdown(
        f'<div class="game-header">'
        f'{logo_img_tag(away, 36)}'
        f'<span style="color:#38bdf8;font-size:20px;font-weight:800;">{away}</span>'
        f'{score_html}'
        f'<span style="color:#475569;font-size:16px;margin:0 8px;">@</span>'
        f'{logo_img_tag(home, 36)}'
        f'<span style="color:#38bdf8;font-size:20px;font-weight:800;">{home}</span>'
        f'<span style="color:#7dd3fc;font-size:12px;margin-left:16px;">'
        f'{w_txt} &nbsp;·&nbsp; Park {pf:.2f}x</span>'
        f'&nbsp;·&nbsp;<span style="color:{status_color};font-size:12px;">{status_tag}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    _all_ids = list(ab_ids) + list(hb_ids)
    _gd = selected_date.strftime('%Y-%m-%d')

    _game_active = status in ('In Progress', 'Manager Challenge', 'Final',
                              'Game Over', 'Completed Early')

    # Show in-progress/final banner but still render the full table (ratings are locked)
    if _game_active:
        _label = '🏁 Final — ratings locked' if is_final else '⚾ Game In Progress — ratings locked before first pitch'
        st.info(_label)

    # Pre-game: only render when both pitchers are confirmed and lineups are official
    _both_pitchers = away_p != 'TBD' and home_p != 'TBD'
    _lineups_ready = _both_pitchers and (game.get('lineups_official') or _game_active)

    if not _lineups_ready:
        _missing = []
        if away_p == 'TBD':
            _missing.append(f'{home} SP')
        if home_p == 'TBD':
            _missing.append(f'{away} SP')
        if not game.get('lineups_official'):
            _missing.append('official lineups')
        st.info(f'⏳ Waiting for: {", ".join(_missing) if _missing else "official lineups"}')
        st.markdown('---')
        continue

    # Recalculate button — clears cached ratings for this game so new weights apply
    if st.button(f'🔄 Recalculate {away} @ {home}', key=f'recalc_{date_key}_{away}_{home}'):
        clear_ratings_for_players(_gd, _all_ids)
        fetch_cache_key_a = f'gv_fetch_{date_key}_{game.get("game_pk","")}_{int(False)}'
        fetch_cache_key_h = f'gv_fetch_{date_key}_{game.get("game_pk","")}_{int(True)}'
        st.session_state.pop(fetch_cache_key_a, None)
        st.session_state.pop(fetch_cache_key_h, None)
        for pid in _all_ids:
            st.session_state.pop(f'locked_{date_key}_{pid}', None)
        st.rerun()

    ac, hc = st.columns(2)
    gk = str(game.get('game_pk', ''))

    with ac:
        st.markdown(f'**{away} Batting** · vs {home_p}')
        if ab_ids:
            render_lineup(ac, ab_ids, a_codes, False, home_pid,
                          home, home, weather, away + ' @ ' + home,
                          home_p, date_key, batter_team=away,
                          game_date=selected_date.strftime('%Y-%m-%d'),
                          event_id=event_id, game_pk=gk)
        else:
            st.info('Lineup pending.')

    with hc:
        st.markdown(f'**{home} Batting** · vs {away_p}')
        if hb_ids:
            render_lineup(hc, hb_ids, h_codes, True, away_pid,
                          away, home, weather, away + ' @ ' + home,
                          away_p, date_key, batter_team=home,
                          game_date=selected_date.strftime('%Y-%m-%d'),
                          event_id=event_id, game_pk=gk)
        else:
            st.info('Lineup pending.')

    st.markdown('---')
