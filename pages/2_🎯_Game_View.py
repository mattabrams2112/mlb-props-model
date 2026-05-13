"""
Game View — both lineups side by side with full HRR ratings per batter.
Rating factors: hit rate, barrel rates, live stadium weather,
ballpark factor, wind, and batting order position (1-5 bonus).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import statsapi
import requests as _req

from feature_engineering import build_features, get_feature_cols, TARGET_COL
from pitcher_data import get_pitcher_season_stats, get_pitcher_name
from statcast_features import get_batter_statcast, get_pitcher_statcast
from weather import get_park_factor
from rating import compute_rating
from lineup_fetcher import get_todays_lineups
from team_logos import get_logo, logo_img_tag
from bvp_stats import get_bvp
from stadium_weather import get_stadium_weather

st.set_page_config(page_title="Game View | MLB Props", page_icon="🎯", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  h1,h2,h3,h4 { color: #38bdf8 !important; }
  .stMarkdown p, label, .stCaption { color: #7dd3fc !important; }
  .stMetric label { color: #38bdf8 !important; }
  .stMetric [data-testid="metric-container"] > div { color: #e0f2fe !important; }
  .weather-box {
    background: #1e293b; border: 1px solid #1e40af;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #7dd3fc; margin-bottom: 8px;
  }
  .game-header {
    background: #1e293b; border: 1px solid #1e40af;
    border-radius: 10px; padding: 12px 18px; margin-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)

MLB_API = 'https://statsapi.mlb.com/api/v1'


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
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


@st.cache_data(show_spinner=False, ttl=3600)
def run_prediction(player_id: int, pitcher_id, is_home: bool, park_team: str,
                   temp_f: float, wind_speed: float, wind_dir: int):
    df = fetch_logs(player_id)
    if df.empty or len(df) < 25:
        return None

    df_feat = build_features(df, fetch_weather=False, override_pitcher_id=pitcher_id)
    idx = df_feat.index[-1]
    df_feat.at[idx, 'is_home']    = int(is_home)
    df_feat.at[idx, 'park_factor'] = get_park_factor(park_team)
    df_feat.at[idx, 'temp_f']     = temp_f
    df_feat.at[idx, 'wind_speed'] = wind_speed
    df_feat.at[idx, 'wind_dir']   = wind_dir

    fc = get_feature_cols()
    dc = df_feat.dropna(subset=fc).reset_index(drop=True)
    if len(dc) < 20:
        return None

    X, y = dc[fc], dc[TARGET_COL]
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for ti, vi in tscv.split(X):
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X.iloc[ti], y.iloc[ti])
        maes.append(mean_absolute_error(y.iloc[vi], m.predict(X.iloc[vi])))

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X, y)

    latest = dc.iloc[-1:].copy()
    latest.at[latest.index[0], 'is_home']    = int(is_home)
    latest.at[latest.index[0], 'park_factor'] = get_park_factor(park_team)
    latest.at[latest.index[0], 'temp_f']     = temp_f
    latest.at[latest.index[0], 'wind_speed'] = wind_speed
    latest.at[latest.index[0], 'wind_dir']   = wind_dir

    proj = max(0.0, float(model.predict(latest[fc])[0]))

    # Rolling stats
    recent_7  = df.tail(7);  r7  = (recent_7['h'] + recent_7['r'] + recent_7['rbi']).mean()
    recent_30 = df.tail(30); r30 = (recent_30['h'] + recent_30['r'] + recent_30['rbi']).mean()
    # Hit rate: rolling BA over last 30 games
    ab30 = recent_30['ab'].sum(); h30 = recent_30['h'].sum()
    ba30 = round(h30 / ab30, 3) if ab30 > 0 else 0.250

    return {
        'proj':    round(proj, 2),
        'mae':     round(float(np.mean(maes)), 3),
        'r7g':     round(float(r7), 2),
        'r30g':    round(float(r30), 2),
        'savg':    round(float(dc['total_season_avg'].iloc[-1]), 2) if not np.isnan(dc['total_season_avg'].iloc[-1]) else 0.0,
        'ba30':    ba30,
        'df':      df,
    }


def build_rating(res, player_id, pitcher_id, park_team, batting_order,
                 temp_f, wind_speed, wind_dir):
    season = int(res['df']['season'].iloc[-1])
    b_sc  = get_batter_statcast(player_id, season)
    p_sc  = get_pitcher_statcast(pitcher_id, season) if pitcher_id else {}
    p_std = get_pitcher_season_stats(pitcher_id, season) if pitcher_id else {}
    bvp   = get_bvp(player_id, pitcher_id) if pitcher_id else {}

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
        bvp_avg           = bvp.get('bvp_avg', 0.250),
        bvp_sample        = bvp.get('bvp_sample', 0),
        batting_order     = batting_order,
        recent_ba         = res['ba30'],
        temp_f            = temp_f,
    ), p_std


def col(v, high, med):
    return '#22c55e' if v >= high else '#eab308' if v >= med else '#ef4444'


def render_side(batter_ids, is_home, opp_pitcher_id, park_team, weather):
    p_std = get_pitcher_season_stats(opp_pitcher_id) if opp_pitcher_id else {}
    p_name = get_pitcher_name(opp_pitcher_id) if opp_pitcher_id else 'TBD'

    # Pitcher info bar
    era  = f"{p_std.get('opp_era',0):.2f}"  if opp_pitcher_id else '—'
    whip = f"{p_std.get('opp_whip',0):.2f}" if opp_pitcher_id else '—'
    kpct = f"{p_std.get('opp_k_pct',0):.1%}" if opp_pitcher_id else '—'
    html  = f'<div style="background:#1e293b;border-radius:6px;padding:7px 12px;margin-bottom:6px;font-size:12px;color:#7dd3fc;">'
    html += f'⚾ <b style="color:#38bdf8;">{p_name}</b> &nbsp;ERA {era} &nbsp;WHIP {whip} &nbsp;K% {kpct}</div>'

    # Table header
    html += '<table style="width:100%;border-collapse:collapse;font-size:12px;">'
    html += '<tr style="background:#0f172a;color:#38bdf8;font-size:11px;font-weight:700;">'
    for h in ['#', '', 'Player', 'Ord', 'Rating', 'Proj', '30g BA', '7g HRR', '30g HRR', 'Barrels']:
        align = 'left' if h in ('', 'Player') else 'center'
        html += f'<th style="padding:5px 7px;text-align:{align};border-bottom:1px solid #1e40af;">{h}</th>'
    html += '</tr>'

    totals = []
    for i, pid in enumerate(batter_ids):
        batting_order = i + 1
        pname, pteam = str(pid), ''
        exc = None
        try:
            info = statsapi.lookup_player(pid)
            if info:
                pname = info[0]['fullName']
                pteam = info[0].get('currentTeam', {}).get('abbreviation', '')
        except Exception as lookup_exc:
            exc = lookup_exc

        res = run_prediction(
            pid, opp_pitcher_id, is_home, park_team,
            weather['temp_f'], weather['wind_speed'], weather['wind_dir_code']
        )

        bg = '#0f172a' if i % 2 == 0 else '#1e293b'
        logo = logo_img_tag(pteam, 18)
        order_color = '#22c55e' if batting_order <= 2 else '#38bdf8' if batting_order <= 5 else '#475569'

        html += f'<tr style="background:{bg};">'
        html += f'<td style="padding:5px 7px;color:#475569;">{batting_order}</td>'
        html += f'<td style="padding:5px 7px;">{logo}</td>'
        html += f'<td style="padding:5px 7px;color:#e0f2fe;font-weight:600;white-space:nowrap;">{pname}</td>'
        html += f'<td style="padding:5px 7px;text-align:center;color:{order_color};font-weight:700;">#{batting_order}</td>'

        if res:
            r, _ = build_rating(res, pid, opp_pitcher_id, park_team,
                                 batting_order, weather['temp_f'],
                                 weather['wind_speed'], weather['wind_dir_code'])
            rc = col(r['total'], 75, 55)
            pc = col(res['proj'], 3.0, 2.0)
            bc = col(res['ba30'], 0.280, 0.250)

            season = int(res['df']['season'].iloc[-1])
            b_sc = get_batter_statcast(pid, season)
            fb_b = b_sc.get('batter_fb_barrel_pct', 0)
            bk_b = b_sc.get('batter_bk_barrel_pct', 0)
            barrel_str = f"FB {fb_b:.1%} BK {bk_b:.1%}"

            html += f'<td style="padding:5px 7px;text-align:center;font-weight:800;color:{rc};">{r["total"]} <span style="font-size:10px;">{r["grade"]}</span></td>'
            html += f'<td style="padding:5px 7px;text-align:center;font-weight:800;font-size:15px;color:{pc};">{res["proj"]}</td>'
            html += f'<td style="padding:5px 7px;text-align:center;color:{bc};">.{int(res["ba30"]*1000):03d}</td>'
            html += f'<td style="padding:5px 7px;text-align:center;color:#7dd3fc;">{res["r7g"]}</td>'
            html += f'<td style="padding:5px 7px;text-align:center;color:#7dd3fc;">{res["r30g"]}</td>'
            html += f'<td style="padding:5px 7px;text-align:center;color:#94a3b8;font-size:11px;">{barrel_str}</td>'
            totals.append((r['total'], res['proj']))
        else:
            html += f'<td colspan="6" style="padding:5px 7px;text-align:center;color:#475569;font-size:11px;">Not enough data</td>'

        html += '</tr>'

    # Totals row
    if totals:
        avg_r  = round(sum(t[0] for t in totals) / len(totals))
        tot_p  = round(sum(t[1] for t in totals), 2)
        rc = col(avg_r, 75, 55); pc = col(tot_p / max(len(totals), 1), 3.0, 2.0)
        html += f'<tr style="background:#0f172a;border-top:2px solid #1e40af;">'
        html += f'<td colspan="3" style="padding:7px;color:#38bdf8;font-weight:700;">LINEUP TOTALS</td>'
        html += f'<td></td>'
        html += f'<td style="padding:7px;text-align:center;font-weight:800;color:{rc};">{avg_r} avg</td>'
        html += f'<td style="padding:7px;text-align:center;font-weight:800;font-size:15px;color:{pc};">{tot_p}</td>'
        html += f'<td colspan="4"></td></tr>'

    html += '</table>'
    return html


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown('## 🎯 Game View — HRR Projections')
st.caption('Rating factors: hit rate · barrel rates · live stadium weather · park factor · wind · batting order')

hdr, btn = st.columns([6, 1])
with btn:
    if st.button('🔄 Refresh', use_container_width=True):
        st.session_state.pop('gv_games', None)
        st.rerun()

if 'gv_games' not in st.session_state:
    with st.spinner('Fetching lineups...'):
        st.session_state['gv_games'] = get_todays_lineups()

games = st.session_state.get('gv_games', [])
if not games:
    st.warning('No games found today.')
    st.stop()

has_lineups = any(g.get('home_batters') or g.get('away_batters') for g in games)
if not has_lineups:
    st.info('Lineups not yet posted. Check back 2–3 hours before first pitch.')
    cols = st.columns(min(len(games), 4))
    for i, g in enumerate(games):
        with cols[i % 4]:
            away_p = get_pitcher_name(g.get('away_pitcher_id')) if g.get('away_pitcher_id') else 'TBD'
            home_p = get_pitcher_name(g.get('home_pitcher_id')) if g.get('home_pitcher_id') else 'TBD'
            w = get_stadium_weather(g.get('home_team', ''), g.get('start_time', ''))
            st.markdown(
                f'<div style="background:#1e293b;border:1px solid #1e40af;border-radius:8px;padding:12px;text-align:center;">'
                f'{logo_img_tag(g.get("away_team",""),28)} <b style="color:#38bdf8;">{g.get("away_team","?")} @ {g.get("home_team","?")}</b> {logo_img_tag(g.get("home_team",""),28)}<br>'
                f'<span style="font-size:11px;color:#7dd3fc;">{away_p} vs {home_p}</span><br>'
                f'<span style="font-size:11px;color:#94a3b8;">{"🏟️ Dome" if w["is_dome"] else str(w["temp_f"]) + "°F · " + w["wind_label"] + " · " + w["condition"]}</span>'
                f'</div>', unsafe_allow_html=True)
    st.stop()

for game in games:
    away = game.get('away_team', '?'); home = game.get('home_team', '?')
    away_pid = game.get('away_pitcher_id'); home_pid = game.get('home_pitcher_id')
    away_p   = get_pitcher_name(away_pid) if away_pid else 'TBD'
    home_p   = get_pitcher_name(home_pid) if home_pid else 'TBD'
    ab_ids   = game.get('away_batters', []); hb_ids = game.get('home_batters', [])

    if not ab_ids and not hb_ids:
        continue

    # Fetch weather forecast at game start time
    game_time_utc = game.get('start_time', '')
    weather = get_stadium_weather(home, game_time_utc)

    # Game header
    pf = get_park_factor(home)
    time_tag = f" at first pitch" if weather.get('game_time_local') and weather['game_time_local'] != 'Now' else ''
    w_str = '🏟️ Dome — weather neutral' if weather['is_dome'] else (
        f"🌡 {weather['temp_f']}°F · 💨 {weather['wind_label']} · {weather['condition']}{time_tag}")
    pf_str = f"Park Factor: {pf:.2f}x"

    st.markdown(
        f'<div class="game-header">'
        f'{logo_img_tag(away, 38)}<span style="color:#38bdf8;font-size:22px;font-weight:800;">{away}</span>'
        f'<span style="color:#475569;font-size:18px;margin:0 10px;">@</span>'
        f'{logo_img_tag(home, 38)}<span style="color:#38bdf8;font-size:22px;font-weight:800;">{home}</span>'
        f'<span style="color:#7dd3fc;font-size:13px;margin-left:20px;">{w_str} &nbsp;·&nbsp; {pf_str}</span>'
        + ('&nbsp;·&nbsp;<span style="color:#22c55e;font-size:12px;">✅ Official lineup</span>'
           if game.get('lineups_official') else
           '&nbsp;·&nbsp;<span style="color:#eab308;font-size:12px;">⏳ Probable pitchers</span>')
        + '</div>',
        unsafe_allow_html=True
    )

    ac, hc = st.columns(2)
    with ac:
        st.markdown(f'**{away} Batting** · vs {home_p}')
        if ab_ids:
            with st.spinner(f'Loading {away}...'):
                st.markdown(render_side(ab_ids, False, home_pid, home, weather),
                            unsafe_allow_html=True)
        else:
            st.info('Lineup pending.')

    with hc:
        st.markdown(f'**{home} Batting** · vs {away_p}')
        if hb_ids:
            with st.spinner(f'Loading {home}...'):
                st.markdown(render_side(hb_ids, True, away_pid, home, weather),
                            unsafe_allow_html=True)
        else:
            st.info('Lineup pending.')

    st.markdown('---')
