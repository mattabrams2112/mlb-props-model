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

# Model training now lives entirely in projection.py — xgboost/lightgbm and the
# feature-engineering helpers are imported there, not here.
from projection import (run_prediction as shared_run_prediction,
                        lineup_context_pct)
# compute_rating's inputs are assembled in exactly one place — see the module
# docstring for what went wrong when this lived in two.
from rating_inputs import score_batter
from lineup_fetcher import get_todays_lineups
from pitcher_data import (get_pitcher_season_stats, get_pitcher_name,
                          get_pitcher_throws, get_pitcher_last_n_starts,
                          get_pitcher_rest_days, get_pitcher_last_pitch_count)
from statcast_features import get_batter_statcast, get_pitcher_statcast
from weather import get_park_factor
from rating import compute_rating
from stadium_weather import get_stadium_weather
from bullpen_data import get_bullpen_stats
from ratings_cache import get_cached_rating, save_rating
from full_tracker import log_play
from team_stats import get_team_recent_scoring, get_team_defense_rating
from umpire_data import get_game_umpire
from data_dir import data_path
from eastern_time import today_str_et

MLB_API      = 'https://statsapi.mlb.com/api/v1'
SEASON       = datetime.now().year
DATABASE_URL = os.environ.get('DATABASE_URL', '')
PREDS_FILE   = data_path('game_preds.csv')
PRED_COLS    = ['date', 'game_id', 'away_team', 'home_team', 'away_pitcher',
                'home_pitcher', 'predicted_winner', 'away_proj', 'home_proj',
                'margin', 'confidence', 'actual_winner', 'result',
                'market_pick', 'pick_ml', 'value_edge']   # must match game_pred_engine.COLS —
                                            # _load_preds slices to this list, so a
                                            # missing col here gets erased on save


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
    today = today_str_et()   # ET — worker runs on a UTC server
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
    home_pf = get_park_factor(home)
    return form_adj + defense_adj + bp_adj + rest_adj + (0.30 * home_pf)  # park-adjusted home field


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


_PRED_CACHE: dict = {}  # (player_id, pitcher_id, date_str, is_home, park_team) → result; cleared at midnight


def _run_prediction(player_id, pitcher_id, is_home, park_team,
                    temp_f, wind_speed, wind_dir, game_date):
    """
    Thin wrapper over the shared pipeline in projection.py.

    The worker used to carry its own copy of this, which is how it ended up
    running a 1.8 ceiling for a week after Game View moved to 2.0. The scoring
    math now lives in exactly one place; this only adds the worker's own
    per-run memo cache and converts the shared 'insufficient_data' exception
    into the None return the calling loop expects.
    """
    date_str  = str(game_date)[:10]
    cache_key = (player_id, pitcher_id, date_str, int(is_home), park_team)
    if cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key]

    try:
        result = shared_run_prediction(
            player_id, pitcher_id, is_home, park_team,
            temp_f, wind_speed, wind_dir, game_date,
            fetch_logs=_fetch_logs)
    except RuntimeError:
        return None      # insufficient history for this batter
    except Exception as e:
        print(f'    Prediction failed for {player_id}: {e}')
        return None

    _PRED_CACHE[cache_key] = result
    return result


def _get_rating(res, pid, pitcher_id, park_team, batting_order,
                temp_f, wind_speed, wind_dir, bp_era, bp_whip,
                is_home, p_std, p_sc, p_last3, p_rest, b_sc,
                team_score, ump_data, opp_defense, game_date=None):
    """
    Thin wrapper over rating_inputs.score_batter — the one place that decides
    what reaches compute_rating. This used to hand-build its own ~80-argument
    call, which is how it ended up passing a different input set than Game View
    for the same player.
    """
    return score_batter(
        res, pid, pitcher_id, park_team, batting_order,
        temp_f, wind_speed, wind_dir,
        bp_era=bp_era, bp_whip=bp_whip, is_home=is_home,
        # The worker has no odds access. Both are label-only in rating.py
        # (Line Edge is displayed, never scored), so the total is unaffected.
        line=None, over_odds=None,
        p_std=p_std, p_sc=p_sc, p_last3=p_last3, p_rest=p_rest, b_sc=b_sc,
        team_score=team_score, ump_data=ump_data, opp_defense=opp_defense,
        pitcher_throws=get_pitcher_throws(pitcher_id) if pitcher_id else 'R',
        game_date=game_date, season=SEASON)





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

    weather  = get_stadium_weather(home, '' if game_started else game.get('start_time', ''),
                                   game_pk=game_pk)
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

        # Prediction and rating run as two passes because lineup context needs
        # every starter's projection before any one batter can be rated —
        # mirroring Game View, which resolves the whole lineup before building
        # rows. Both passes stay pooled; the expensive work is model training.
        def predict_batter(pid):
            try:
                ocode      = batter_codes.get(int(pid), 0)
                is_starter = (ocode % 100 == 0) and (ocode > 0)
                spot       = ocode // 100
                if not is_starter or spot == 0:
                    return None

                # Already cached — use frozen rating
                cached = get_cached_rating(game_date, pid)
                if cached:
                    return ('cached', spot, cached)

                # Game already started and no cache — skip
                if game_started:
                    return None

                res = _run_prediction(pid, opp_pid, is_home, park_team,
                                      temp_f, wind_sp, wind_dr, game_date)
                if not res:
                    return None
                return ('new', spot, res)
            except Exception as e:
                print(f'    Error predicting player {pid}: {e}')
                return None

        with ThreadPoolExecutor(max_workers=8) as exe:
            preds = list(exe.map(predict_batter, batter_ids))

        # Lineup context — how strong is the batting order around this player.
        # The calculation itself lives in projection.lineup_context_pct.
        #
        # Cached batters are included via their frozen projection. They skip
        # _run_prediction entirely, so on the worker's second and later passes
        # of the day most of the lineup is cached — keying this off fresh
        # predictions alone would average one or two batters and hand everyone
        # else a garbage context. The frozen number is post-calibration where a
        # fresh one is pre-, so it is a proxy rather than a match, but a close
        # proxy for all nine beats an exact figure for two.
        _starter_projs = {}
        for _p in preds:
            if not _p:
                continue
            _kind, _sp, _payload = _p
            if _kind == 'new':
                _starter_projs[_sp] = _payload['proj']
            else:
                _starter_projs[_sp] = _payload[2]   # frozen (rating, grade, proj)

        def _ctx_pct_for(spot):
            return lineup_context_pct(spot, _starter_projs)

        def rate_batter(item):
            pid, pred = item
            if pred is None:
                return None
            kind, spot, payload = pred
            try:
                if kind == 'cached':
                    locked_rating, locked_grade, locked_proj = payload
                    print(f'    [cached] {pid} — Rating {locked_rating} · Proj {locked_proj}')
                    return (locked_rating, locked_proj, None, None, None, None, None, None)

                res  = payload
                b_sc = get_batter_statcast(pid, SEASON)

                def _rate(p):
                    _ctx = dict(res)
                    _ctx['proj'] = p
                    return _get_rating(_ctx, pid, opp_pid, park_team, spot,
                                       temp_f, wind_sp, wind_dr,
                                       bp_era, bp_whip, is_home,
                                       p_std, p_sc, p_last3, p_rest, b_sc,
                                       team_sc, ump_data, opp_def,
                                       game_date=game_date)

                # Two passes, matching Game View: rate the context-adjusted
                # projection, use THAT rating to pick the calibration factor,
                # then re-rate the corrected projection. The worker previously
                # rated the raw projection and separately logged a calibrated
                # one, so the rating and the projection beside it described
                # different numbers.
                base_proj = round(max(0.5, res['proj'] * (1 + _ctx_pct_for(spot))), 2)
                _pass1    = _rate(base_proj)
                try:
                    from calibration import get_correction_factor
                    _calib = get_correction_factor(_pass1['total'])
                except Exception:
                    _calib = 1.0
                proj = max(0.5, base_proj * _calib * (1 + _pass1.get('matchup_pct', 0.0)))
                # Re-cap at the player's realistic ceiling so stacked
                # multipliers can't manufacture a 5+ HRR projection
                proj   = round(min(proj, res.get('proj_ceiling', proj)), 2)
                r_data = _rate(proj)

                # Look up player name
                try:
                    info  = statsapi.lookup_player(pid)
                    pname = info[0]['fullName'] if info else str(pid)
                except Exception:
                    pname = str(pid)

                return (r_data['total'], proj, pname, r_data['grade'], r_data,
                        base_proj, res.get('proj_ceiling'), res.get('r30g'))
            except Exception as e:
                print(f'    Error rating player {pid}: {e}')
                return None

        with ThreadPoolExecutor(max_workers=8) as exe:
            results = list(exe.map(rate_batter, zip(batter_ids, preds)))

        for pid, result in zip(batter_ids, results):
            if result is None:
                continue
            rating, proj, pname, grade, r_data, base_proj, proj_ceiling, r30g = result

            totals.append((rating, proj))

            if pname is None:
                continue  # cached — already saved

            # Save to ratings cache (freezes the rating pre-game)
            save_rating(game_date, pid, rating, grade, proj,
                        player_name=pname, team=batter_team, vs_pitcher=opp_p_name)

            # Log to full play log
            log_play(player=pname, team=batter_team,
                     rating=rating, grade=grade, projected=proj,
                     base_proj=base_proj, proj_ceiling=proj_ceiling,
                     # Tag the source: worker and Game View reach base_proj by
                     # the same route now, but a calibration fit must still be
                     # able to prove that rather than assume it.
                     proj_src='worker',
                     r30g=r30g,
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
    # ET, not datetime.now() — the worker runs on a UTC server, and using UTC
    # here stamps evening-ET games (past 8pm ET = next UTC day) with tomorrow's
    # date, so they land under the wrong day and never resolve in Daily Results.
    game_date = today_str_et()
    date_str  = datetime.strptime(game_date, '%Y-%m-%d').strftime('%m/%d/%Y')

    # Clear stale predictions from previous days
    stale = [k for k in _PRED_CACHE if k[2] != game_date]
    for k in stale:
        del _PRED_CACHE[k]

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
