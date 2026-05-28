"""
Background worker — runs on Railway cron every 30 min (9 AM – 11:30 PM).
Replicates Game View logic without Streamlit:
  - Fetches today's lineups
  - Calculates and freezes player ratings
  - Logs qualifying plays to full_play_log
  - Saves game predictions
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests as _req
import statsapi
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False

from feature_engineering import build_features, get_feature_cols, TARGET_COL
from lineup_fetcher import get_todays_lineups
from pitcher_data import (get_pitcher_season_stats, get_pitcher_name,
                          get_pitcher_throws, get_pitcher_last_n_starts,
                          get_pitcher_rest_days)
from statcast_features import get_batter_statcast, get_pitcher_statcast
from weather import get_park_factor
from rating import compute_rating
from bvp_stats import get_bvp
from stadium_weather import get_stadium_weather
from bullpen_data import get_bullpen_stats
from ratings_cache import get_cached_rating, save_rating
from full_tracker import log_play
from team_stats import get_team_recent_scoring, get_team_defense_rating
from umpire_data import get_game_umpire
from data_dir import data_path

MLB_API      = 'https://statsapi.mlb.com/api/v1'
SEASON       = datetime.now().year
DATABASE_URL = os.environ.get('DATABASE_URL', '')
PREDS_FILE   = data_path('game_preds.csv')
PRED_COLS    = ['date', 'game_id', 'away_team', 'home_team', 'away_pitcher',
                'home_pitcher', 'predicted_winner', 'away_proj', 'home_proj',
                'margin', 'confidence', 'actual_winner', 'result']


# ── DB helpers (mirrors Game Predictions page) ────────────────────────────────

def _get_engine():
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine
        url = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        if '?' not in url:
            url += '?sslmode=require'
        elif 'sslmode' not in url:
            url += '&sslmode=require'
        return create_engine(url, connect_args={'connect_timeout': 10})
    except Exception:
        return None


def _load_preds() -> pd.DataFrame:
    engine = _get_engine()
    if engine:
        try:
            df = pd.read_sql('SELECT * FROM game_predictions ORDER BY date DESC', engine)
            for c in PRED_COLS:
                if c not in df.columns:
                    df[c] = ''
            return df[PRED_COLS]
        except Exception:
            pass
    if os.path.exists(PREDS_FILE):
        try:
            return pd.read_csv(PREDS_FILE, dtype=str).fillna('')
        except Exception:
            pass
    return pd.DataFrame(columns=PRED_COLS)


def _save_preds(df: pd.DataFrame):
    engine = _get_engine()
    if engine:
        try:
            df.to_sql('game_predictions', engine, if_exists='replace', index=False)
            return
        except Exception:
            pass
    df.to_csv(PREDS_FILE, index=False)


def _add_game_pred(row: dict, game_date: str, game_started: bool = False):
    df    = _load_preds()
    today = datetime.now().strftime('%Y-%m-%d')
    match = (not df.empty and
             (df['game_id'].astype(str) == str(row['game_id'])) &
             (df['date'].astype(str).str[:10] == game_date))
    if match.any():
        if not game_started and game_date >= today:
            idx = df[match].index[0]
            for c in ['predicted_winner', 'away_proj', 'home_proj', 'margin',
                      'confidence', 'away_pitcher', 'home_pitcher']:
                if c in row:
                    df.at[idx, c] = row[c]
            _save_preds(df)
        return
    new = pd.DataFrame([{c: row.get(c, '') for c in PRED_COLS}])
    df  = pd.concat([df, new], ignore_index=True)
    _save_preds(df)


# ── Prediction helpers ────────────────────────────────────────────────────────

def _margin_to_confidence(margin):
    a = abs(margin)
    if a >= 4.0:  return 'Strong'
    if a >= 2.0:  return 'Moderate'
    if a >= 0.75: return 'Lean'
    return 'Toss-up'


def _get_adjustments(home, away, home_pid, away_pid, game_date):
    home_sc = get_team_recent_scoring(home)
    away_sc = get_team_recent_scoring(away)
    home_rd = home_sc.get('team_runs_avg', 4.5) - home_sc.get('team_runs_allowed_avg', 4.5)
    away_rd = away_sc.get('team_runs_avg', 4.5) - away_sc.get('team_runs_allowed_avg', 4.5)
    form_adj    = round((home_rd - away_rd) * 0.20, 2)
    home_def    = get_team_defense_rating(home, SEASON).get('def_rating', 0.0)
    away_def    = get_team_defense_rating(away, SEASON).get('def_rating', 0.0)
    defense_adj = round((away_def - home_def) * 0.15, 2)
    home_bp_era = get_bullpen_stats(home, SEASON).get('bp_era', 4.20)
    away_bp_era = get_bullpen_stats(away, SEASON).get('bp_era', 4.20)
    bp_adj      = round((away_bp_era - home_bp_era) * 0.12, 2)
    home_rest   = get_pitcher_rest_days(home_pid, SEASON, game_date).get('rest_factor', 0.0) if home_pid else 0.0
    away_rest   = get_pitcher_rest_days(away_pid, SEASON, game_date).get('rest_factor', 0.0) if away_pid else 0.0
    rest_adj    = round((home_rest - away_rest) * 0.15, 2)
    return form_adj + defense_adj + bp_adj + rest_adj + 0.30  # 0.30 = home field


def _formula_prediction(home, away, home_pid, away_pid, game_date):
    base = 4.50
    hp   = get_pitcher_season_stats(home_pid) if home_pid else {}
    ap   = get_pitcher_season_stats(away_pid) if away_pid else {}
    h_era = hp.get('opp_era', 4.50); h_fip = hp.get('opp_fip', h_era)
    a_era = ap.get('opp_era', 4.50); a_fip = ap.get('opp_fip', a_era)
    home_pq = 0.55 * h_era + 0.45 * h_fip
    away_pq = 0.55 * a_era + 0.45 * a_fip
    park    = get_park_factor(home)
    ht      = get_team_recent_scoring(home)
    at      = get_team_recent_scoring(away)
    away_proj = base * (home_pq / 4.50) * (at.get('team_runs_avg', 4.5) / 4.50) * (ht.get('team_runs_allowed_avg', 4.5) / 4.50) * park
    home_proj = base * (away_pq / 4.50) * (ht.get('team_runs_avg', 4.5) / 4.50) * (at.get('team_runs_allowed_avg', 4.5) / 4.50) * park
    total_adj = _get_adjustments(home, away, home_pid, away_pid, game_date)
    home_proj += total_adj
    away_proj  = round(min(max(away_proj, 1.5), 15.0), 1)
    home_proj  = round(min(max(home_proj, 1.5), 15.0), 1)
    margin     = round(home_proj - away_proj, 1)
    return home if margin >= 0 else away, away_proj, home_proj, margin


# ── Player prediction (mirrors run_prediction in Game View) ───────────────────

def _fetch_logs(player_id: int) -> pd.DataFrame:
    from game_log_fetcher import fetch_player_logs
    return fetch_player_logs(player_id)


def _run_prediction(player_id, pitcher_id, is_home, park_team,
                    temp_f, wind_speed, wind_dir, game_date):
    df = _fetch_logs(player_id)
    if df.empty or len(df) < 25:
        return None
    try:
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
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        lgb.fit(X, y)

    latest = dc.iloc[-1:].copy()
    latest.at[latest.index[0], 'is_home']     = int(is_home)
    latest.at[latest.index[0], 'park_factor'] = get_park_factor(park_team)
    latest.at[latest.index[0], 'temp_f']      = temp_f
    latest.at[latest.index[0], 'wind_speed']  = wind_speed
    latest.at[latest.index[0], 'wind_dir']    = wind_dir

    latest_X = latest[fc].apply(pd.to_numeric, errors='coerce').fillna(0)
    xgb_pred = float(xgb.predict(latest_X)[0])
    proj = max(0.0, (xgb_pred * 0.55 + float(lgb.predict(latest_X)[0]) * 0.45)) if HAS_LGBM else max(0.0, xgb_pred)

    season_avg = float(dc['total_season_avg'].iloc[-1]) if not np.isnan(dc['total_season_avg'].iloc[-1]) else 0
    r30_avg    = float((df.tail(30)['h'] + df.tail(30)['r'] + df.tail(30)['rbi']).mean())
    proj       = max(proj, max(season_avg * 0.30, r30_avg * 0.30))
    proj       = min(proj, min(3.5, max(r30_avg * 1.5, season_avg * 1.5, 1.5)))

    r7  = df.tail(7);  hrr7  = (r7['h']  + r7['r']  + r7['rbi']).mean()
    r30 = df.tail(30); hrr30 = (r30['h'] + r30['r'] + r30['rbi']).mean()
    ab30 = r30['ab'].sum(); h30 = r30['h'].sum()
    hg   = df[df['is_home'] == 1]; ag = df[df['is_home'] == 0]

    return {
        'proj':       round(proj, 2),
        'r7g':        round(float(hrr7), 2),
        'r30g':       round(float(hrr30), 2),
        'savg':       round(season_avg, 2),
        'ba30':       round(h30 / ab30, 3) if ab30 > 0 else 0.250,
        'home_hrr':   round(float((hg['h'] + hg['r'] + hg['rbi']).mean()), 2) if len(hg) >= 5 else None,
        'away_hrr':   round(float((ag['h'] + ag['r'] + ag['rbi']).mean()), 2) if len(ag) >= 5 else None,
        'r20g_venue': round(float(dc['hrr_20g_venue'].iloc[-1]), 2)
                      if 'hrr_20g_venue' in dc.columns and not np.isnan(dc['hrr_20g_venue'].iloc[-1]) else None,
        'ba_venue':   round(float(dc['ba_20g_venue'].iloc[-1]), 3)
                      if 'ba_20g_venue' in dc.columns and not np.isnan(dc['ba_20g_venue'].iloc[-1]) else None,
        'df':         df,
    }


def _get_rating(res, pid, pitcher_id, park_team, batting_order,
                temp_f, wind_speed, wind_dir, bp_era, bp_whip,
                is_home, p_std, p_sc, p_last3, p_rest, b_sc,
                team_score, ump_data, opp_defense):
    bvp = get_bvp(pid, pitcher_id) if pitcher_id else {}
    return compute_rating(
        recent_7g             = res['r7g'],
        recent_30g            = res['r30g'],
        season_avg            = res['savg'],
        opp_era               = p_std.get('opp_era', 4.30),
        opp_whip              = p_std.get('opp_whip', 1.28),
        batter_fb_barrel      = b_sc.get('batter_fb_barrel_pct', 0.080),
        batter_bk_barrel      = b_sc.get('batter_bk_barrel_pct', 0.040),
        batter_os_barrel      = b_sc.get('batter_os_barrel_pct', 0.050),
        pitcher_fb_barrel     = p_sc.get('pitcher_fb_barrel_pct', 0.080),
        pitcher_bk_barrel     = p_sc.get('pitcher_bk_barrel_pct', 0.040),
        pitcher_os_barrel     = p_sc.get('pitcher_os_barrel_pct', 0.050),
        batter_fb_seen        = b_sc.get('batter_fb_seen_pct', 0.55),
        batter_bk_seen        = b_sc.get('batter_bk_seen_pct', 0.25),
        batter_os_seen        = b_sc.get('batter_os_seen_pct', 0.20),
        park_factor           = get_park_factor(park_team),
        wind_speed            = wind_speed,
        wind_dir              = wind_dir,
        bvp_avg               = bvp.get('bvp_avg', 0.250),
        bvp_sample            = bvp.get('bvp_sample', 0),
        batting_order         = batting_order,
        recent_ba             = res['ba30'],
        temp_f                = temp_f,
        projection            = res['proj'],
        bp_era                = bp_era,
        bp_whip               = bp_whip,
        line                  = None,
        over_odds             = None,
        home_hrr              = res.get('home_hrr'),
        away_hrr              = res.get('away_hrr'),
        is_home               = is_home,
        recent_20g            = res.get('r20g_venue'),
        recent_ba_venue       = res.get('ba_venue'),
        batter_hard_hit_pct   = b_sc.get('batter_hard_hit_pct', 0.360),
        pitcher_hard_hit_pct  = p_sc.get('pitcher_hard_hit_pct', 0.360),
        batter_xba            = b_sc.get('batter_xba', 0.250),
        pitcher_xba_allowed   = p_sc.get('pitcher_xba_allowed', 0.250),
        batter_avg_ev         = b_sc.get('batter_avg_ev', 88.0),
        pitcher_avg_ev        = p_sc.get('pitcher_avg_ev', 88.0),
        opp_fip               = p_std.get('opp_fip', 4.20),
        opp_last3_era         = p_last3.get('opp_last3_era', 4.30),
        opp_last3_whip        = p_last3.get('opp_last3_whip', 1.28),
        pitcher_throws        = get_pitcher_throws(pitcher_id) if pitcher_id else 'R',
        batter_xba_vs_rhp     = b_sc.get('batter_xba_vs_rhp', 0.250),
        batter_xba_vs_lhp     = b_sc.get('batter_xba_vs_lhp', 0.250),
        batter_hard_hit_vs_rhp= b_sc.get('batter_hard_hit_vs_rhp', 0.360),
        batter_hard_hit_vs_lhp= b_sc.get('batter_hard_hit_vs_lhp', 0.360),
        team_runs_avg         = team_score.get('team_runs_avg', 4.5),
        umpire_tendency       = ump_data.get('umpire_tendency', 0.0),
        opp_def_rating        = opp_defense.get('def_rating', 0.0),
        pitcher_rest_factor   = p_rest.get('rest_factor', 0.0),
        pitcher_gb_pct        = p_sc.get('pitcher_gb_pct', 0.430),
        pitcher_fb_thrown    = p_sc.get('pitcher_fb_thrown_pct',   0.55),
        pitcher_bk_thrown    = p_sc.get('pitcher_bk_thrown_pct',   0.25),
        pitcher_os_thrown    = p_sc.get('pitcher_os_thrown_pct',   0.20),
        batter_whiff_pct_fb  = b_sc.get('batter_whiff_pct_fb',    0.200),
        batter_whiff_pct_bk  = b_sc.get('batter_whiff_pct_bk',    0.330),
        batter_whiff_pct_os  = b_sc.get('batter_whiff_pct_os',    0.310),
        pitcher_whiff_pct_fb = p_sc.get('pitcher_whiff_pct_fb',   0.200),
        pitcher_whiff_pct_bk = p_sc.get('pitcher_whiff_pct_bk',   0.330),
        pitcher_whiff_pct_os = p_sc.get('pitcher_whiff_pct_os',   0.310),
        opp_k_pct            = p_sc.get('pitcher_k_pct',          0.222),
        opp_bb_pct           = p_sc.get('pitcher_bb_pct',          0.083),
        opp_babip            = p_sc.get('pitcher_babip',           0.300),
        opp_whiff_pct        = p_sc.get('pitcher_whiff_pct',       0.245),
        opp_k_pct_vs_lhb   = p_sc.get('pitcher_k_pct_vs_lhb',    None),
        opp_k_pct_vs_rhb   = p_sc.get('pitcher_k_pct_vs_rhb',    None),
        opp_babip_vs_lhb   = p_sc.get('pitcher_babip_vs_lhb',    None),
        opp_babip_vs_rhb   = p_sc.get('pitcher_babip_vs_rhb',    None),
        batter_k_pct        = b_sc.get('batter_k_pct',            0.222),
        batter_bb_pct       = b_sc.get('batter_bb_pct',           0.083),
        batter_babip        = b_sc.get('batter_babip',            0.300),
        batter_whiff_pct    = b_sc.get('batter_whiff_pct',        0.245),
        batter_k_pct_vs_rhp = b_sc.get('batter_k_pct_vs_rhp',    None),
        batter_k_pct_vs_lhp = b_sc.get('batter_k_pct_vs_lhp',    None),
        batter_babip_vs_rhp = b_sc.get('batter_babip_vs_rhp',    None),
        batter_babip_vs_lhp = b_sc.get('batter_babip_vs_lhp',    None),
    )


# ── Process one game ──────────────────────────────────────────────────────────

def process_game(game, game_date):
    home     = game.get('home_team', '')
    away     = game.get('away_team', '')
    home_pid = game.get('home_pitcher_id')
    away_pid = game.get('away_pitcher_id')
    status   = game.get('status', '')
    game_pk  = str(game.get('game_pk', ''))
    pre_game = status in ('Preview', 'Pre-Game', 'Scheduled', 'Warmup', '')
    game_started = not pre_game

    weather  = get_stadium_weather(home, '' if game_started else game.get('start_time', ''))
    temp_f   = weather.get('temp_f', 72)
    wind_sp  = weather.get('wind_speed', 5)
    wind_dr  = weather.get('wind_dir_code', 0)

    try:
        ump_data = get_game_umpire(int(game_pk)) if game_pk else {}
    except Exception:
        ump_data = {}

    home_hrr_total = 0.0
    away_hrr_total = 0.0

    sides = [
        (game.get('home_batters', []), game.get('home_batter_codes', {}), away_pid, home, home, True),
        (game.get('away_batters', []), game.get('away_batter_codes', {}), home_pid, home, away, False),
    ]

    for batter_ids, batter_codes, opp_pid, park_team, batter_team, is_home in sides:
        if not batter_ids:
            continue

        p_std    = get_pitcher_season_stats(opp_pid, SEASON) if opp_pid else {}
        p_sc     = get_pitcher_statcast(opp_pid, SEASON)     if opp_pid else {}
        p_last3  = get_pitcher_last_n_starts(opp_pid, 3, SEASON) if opp_pid else {}
        p_rest   = get_pitcher_rest_days(opp_pid, SEASON, game_date) if opp_pid else {}
        bp       = get_bullpen_stats(batter_team, SEASON)
        bp_era   = bp.get('bp_era', 4.20)
        bp_whip  = bp.get('bp_whip', 1.30)
        team_sc  = get_team_recent_scoring(batter_team)
        opp_team = away if is_home else home
        opp_def  = get_team_defense_rating(opp_team, SEASON)
        opp_p_name = get_pitcher_name(opp_pid) if opp_pid else 'TBD'

        totals = []

        def process_batter(pid):
            try:
                ocode      = batter_codes.get(int(pid), 0)
                is_starter = (ocode % 100 == 0) and (ocode > 0)
                spot       = ocode // 100
                if not is_starter or spot == 0:
                    return None

                # Already cached — use frozen rating
                cached = get_cached_rating(game_date, pid)
                if cached:
                    locked_rating, locked_grade, locked_proj = cached
                    print(f'    [cached] {pid} — Rating {locked_rating} · Proj {locked_proj}')
                    return (locked_rating, locked_proj, None, None, None)

                # Game already started and no cache — skip
                if game_started:
                    return None

                res = _run_prediction(pid, opp_pid, is_home, park_team,
                                      temp_f, wind_sp, wind_dr, game_date)
                if not res:
                    return None

                b_sc   = get_batter_statcast(pid, SEASON)
                r_data = _get_rating(res, pid, opp_pid, park_team, spot,
                                     temp_f, wind_sp, wind_dr,
                                     bp_era, bp_whip, is_home,
                                     p_std, p_sc, p_last3, p_rest, b_sc,
                                     team_sc, ump_data, opp_def)

                rating = r_data['total']
                grade  = r_data['grade']
                proj   = round(res['proj'], 2)

                # Look up player name
                try:
                    info  = statsapi.lookup_player(pid)
                    pname = info[0]['fullName'] if info else str(pid)
                except Exception:
                    pname = str(pid)

                return (rating, proj, pname, grade, r_data)
            except Exception as e:
                print(f'    Error on player {pid}: {e}')
                return None

        with ThreadPoolExecutor(max_workers=3) as exe:
            results = list(exe.map(process_batter, batter_ids))

        for pid, result in zip(batter_ids, results):
            if result is None:
                continue
            rating, proj, pname, grade, r_data = result

            totals.append((rating, proj))

            if pname is None:
                continue  # cached — already saved

            # Save to ratings cache (freezes the rating pre-game)
            save_rating(game_date, pid, rating, grade, proj,
                        player_name=pname, team=batter_team, vs_pitcher=opp_p_name)

            # Log to full play log
            log_play(player=pname, team=batter_team,
                     rating=rating, grade=grade, projected=proj,
                     vs_pitcher=opp_p_name, is_home=is_home,
                     game_date=game_date, game_started=False)

            print(f'    {pname} ({batter_team}) — {rating} {grade} · Proj {proj}')

        team_proj = sum(p for _, p in totals)
        if is_home:
            home_hrr_total = team_proj
        else:
            away_hrr_total = team_proj

    return home_hrr_total, away_hrr_total


# ── Save game prediction ──────────────────────────────────────────────────────

def save_prediction(game, home_hrr, away_hrr, game_date):
    home     = game.get('home_team', '')
    away     = game.get('away_team', '')
    home_pid = game.get('home_pitcher_id')
    away_pid = game.get('away_pitcher_id')
    status   = game.get('status', '')
    gid      = f'{away}_{home}'
    game_started = status not in ('Preview', 'Pre-Game', 'Scheduled', 'Warmup', '')

    home_p = get_pitcher_name(home_pid) if home_pid else 'TBD'
    away_p = get_pitcher_name(away_pid) if away_pid else 'TBD'

    if home_hrr > 0 and away_hrr > 0:
        total_adj = _get_adjustments(home, away, home_pid, away_pid, game_date)
        adj_home  = round(home_hrr + total_adj, 1)
        adj_away  = round(away_hrr, 1)
        margin    = round(adj_home - adj_away, 1)
        winner    = home if margin >= 0 else away
        away_proj, home_proj = adj_away, adj_home
    else:
        winner, away_proj, home_proj, margin = _formula_prediction(
            home, away, home_pid, away_pid, game_date)

    confidence = _margin_to_confidence(margin)

    _add_game_pred({
        'game_id':          gid,
        'date':             game_date,
        'away_team':        away,
        'home_team':        home,
        'away_pitcher':     away_p,
        'home_pitcher':     home_p,
        'predicted_winner': winner,
        'away_proj':        away_proj,
        'home_proj':        home_proj,
        'margin':           margin,
        'confidence':       confidence,
        'actual_winner':    '',
        'result':           '',
    }, game_date, game_started=game_started)

    print(f'  → Prediction: {winner} wins ({confidence}, margin {abs(margin):.1f})')


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    game_date = datetime.now().strftime('%Y-%m-%d')
    date_str  = datetime.now().strftime('%m/%d/%Y')

    print(f'\n=== Worker {datetime.now().strftime("%Y-%m-%d %H:%M")} ===')

    try:
        games = get_todays_lineups(date_str)
    except Exception as e:
        print(f'Failed to fetch lineups: {e}')
        return

    if not games:
        print('No games today.')
        return

    has_lineups = any(g.get('home_batters') or g.get('away_batters') for g in games)
    if not has_lineups:
        print('Lineups not posted yet.')
        return

    print(f'{len(games)} games found.')

    for game in games:
        home = game.get('home_team', '?')
        away = game.get('away_team', '?')
        print(f'\n{away} @ {home}  [{game.get("status", "")}]')
        try:
            home_hrr, away_hrr = process_game(game, game_date)
            save_prediction(game, home_hrr, away_hrr, game_date)
        except Exception as e:
            print(f'  Error: {e}')

    print('\nDone.')


if __name__ == '__main__':
    run()
