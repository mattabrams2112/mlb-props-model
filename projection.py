"""
The HRR projection pipeline — the single source of truth.

This used to exist twice: `run_prediction` in Game View and `_run_prediction` in
worker.py. Both scored every batter, both wrote `ratings_cache`, and whichever
reached a player first froze that player's rating permanently (`save_rating` is
first-write-wins and Game View honors the frozen value). The two copies drifted,
so the rating you bet depended on which program got there first:

  * The 2026-08-03 ceiling loosening (1.8 -> 2.0) was only ever applied to the
    Game View copy. For every play the worker rated first it silently never
    happened — and the worker runs on a 30-minute cron, so that was most of
    them. Twelve days of evaluation measured a blend of two models.
  * The worker had no pitcher multiplier and no lineup-context adjustment.
  * The worker trained on real pitcher features with recency weights and
    batter-type hyperparameters; Game View did none of that.

Anything changed here now changes both callers at once. Do not reintroduce a
second copy — if a caller needs different behavior, it belongs in this file
behind an argument, where the difference is visible.

CONFIGURATION — the union of both old pipelines, keeping the worker's modelling.
Game View used to run `fast_mode=True`, which fills every historical pitcher
column with league averages (zero variance, pure noise), so it had to train with
`include_pitcher=False` and recover pitcher quality afterward via a multiplier.
The worker ran `fast_mode=False`, looking up the real starting pitcher per
historical game, so its model could actually learn batter-vs-pitcher-quality.
The merged pipeline keeps the worker's version — real pitcher history, pitcher
features in training, recency weighting and power/contact tuning — and Game View
inherits all four.

That makes the post-hoc pitcher multiplier redundant: with real pitcher features
in training, applying it as well counts the opposing starter twice, on top of
the five rating components that already read pitcher (Starter Matchup, Pitcher
Context, Batted Ball Edge, Bullpen, Platoon). The 2026-08-05 diagnostic already
measured the model over-responding to pitcher by ~2x —
corr(opp_era, projected) 0.263 vs corr(opp_era, actual) 0.142 — so it is
switched OFF below. Flip APPLY_PITCHER_MULTIPLIER to re-enable it; that is the
single constant, and the pitcher_multiplier function is kept intact for it.

Cost of `fast_mode=False`: the first projection for a batter looks up the
starting pitcher for every game in their history. Those lookups are cached to
disk (cache_game_pitchers.csv) and pitcher logs in memory, so it amortizes, but
a cold cache after a deploy makes the first Game View load noticeably slower.
"""
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from feature_engineering import build_features, get_feature_cols, TARGET_COL
from weather import get_park_factor

# Projection ceiling as a multiple of the batter's own recent form. Loosened
# 1.8 -> 2.0 on 2026-08-03: at 1.8 the cap starved the top of the scale (only 6
# plays at 85-89 and ZERO at 95+ in 12 days), leaving no volume to measure. 2.0
# still kills the fantasy projections (a 1.2-form hitter caps at 2.4, not 5.0)
# and keeps Edge honest; it only admits borderline plays back into the band.
# This has always been a volume/quality knob, never a fitted number.
PROJ_CEILING_MULT = 2.0

# Post-hoc pitcher multiplier. OFF because the model now trains on real pitcher
# features — see the module docstring. Set True to restore Game View's old
# behavior, accepting that it double-counts the opposing starter.
APPLY_PITCHER_MULTIPLIER = False

# League baselines for the pitcher multiplier
_L_ERA, _L_FIP, _L_WHIP, _L_K, _L_BB = 4.30, 4.20, 1.28, 0.222, 0.083

# Lineup context
LEAGUE_AVG_HRR = 1.8


def _default_fetch_logs(player_id: int) -> pd.DataFrame:
    from game_log_fetcher import fetch_player_logs
    return fetch_player_logs(player_id)


def pitcher_multiplier(row) -> float:
    """
    Post-hoc opposing-starter adjustment, batter-favorable above 1.0.

    XGBoost cannot learn pitcher quality from the training data (see the module
    docstring), so it is applied to the finished projection instead. Weights
    without BvP: ERA 28%, FIP 22%, WHIP 20%, K% 18%, BB% 12%. With a usable BvP
    sample the pitcher stats scale to 83% and BvP takes the remaining 17%.
    Clamped to [0.75, 1.25] so one bad stat can't run away with the projection.
    """
    try:
        def _sf(col, default, floor_v=None):
            try:
                v = float(row.get(col, default))
            except (TypeError, ValueError):
                v = default
            if v <= 0:
                v = default
            return max(v, floor_v) if floor_v is not None else v

        era_f  = _sf('opp_era',    _L_ERA)  / _L_ERA   # high ERA  -> batter +
        fip_f  = _sf('opp_fip',    _L_FIP)  / _L_FIP   # high FIP  -> batter +
        whip_f = _sf('opp_whip',   _L_WHIP) / _L_WHIP  # high WHIP -> batter +
        k_f    = _L_K / _sf('opp_k_pct', _L_K, 0.10)   # high K%   -> batter -
        bb_f   = _sf('opp_bb_pct', _L_BB)  / _L_BB     # high BB%  -> batter +

        if int(row.get('bvp_sample', 0) or 0):
            bvp_f = _sf('bvp_avg', 0.250) / 0.250
            mult = (0.2324 * era_f + 0.1826 * fip_f + 0.1660 * whip_f +
                    0.1494 * k_f   + 0.0996 * bb_f  + 0.1700 * bvp_f)
        else:
            mult = (0.28 * era_f + 0.22 * fip_f + 0.20 * whip_f +
                    0.18 * k_f   + 0.12 * bb_f)

        return min(1.25, max(0.75, mult))
    except Exception:
        return 1.0


def lineup_context_pct(spot: int, starter_projs: dict) -> float:
    """
    How strong is the batting order around this player, as a +/-12% adjustment.

    `starter_projs` maps batting-order spot -> projection for the whole lineup.
    Below 5 known spots the average describes nothing, so it returns neutral
    rather than inventing a swing off one or two batters.
    """
    if len(starter_projs) < 5 or spot not in starter_projs:
        return 0.0
    team_avg = sum(starter_projs.values()) / len(starter_projs)
    prev_s   = 9 if spot == 1 else spot - 1
    next_s   = 1 if spot == 9 else spot + 1
    nbrs     = [starter_projs[s] for s in (prev_s, next_s) if s in starter_projs]
    nbr_avg  = sum(nbrs) / len(nbrs) if nbrs else team_avg
    ctx_avg  = 0.5 * team_avg + 0.5 * nbr_avg
    return max(-0.12, min(0.12, (ctx_avg - LEAGUE_AVG_HRR) / LEAGUE_AVG_HRR * 0.4))


# The one authoritative eligibility rule: build the features, then require this
# many rows that survive `dropna` on the full feature set.
MIN_USABLE_ROWS = 20

# Cheap early-out so we don't build features for an obviously empty history.
# This is a SPEED guard, not the eligibility rule. Passing it does not mean a
# batter can be projected — rolling windows (30g at min_periods=15) and pitcher
# features drop rows that no raw game count can predict. Never report this
# number as "the requirement"; report the usable-row count instead.
PREFILTER_GAMES = 25


class InsufficientData(RuntimeError):
    """Not enough history to project, carrying the counts so callers can say why.

    Subclasses RuntimeError so existing broad handlers (worker.py's
    `except RuntimeError: return None`) keep working unchanged.
    """

    def __init__(self, games, usable=None):
        self.games    = games
        self.usable   = usable
        self.required = MIN_USABLE_ROWS
        if usable is None:
            # Failed the cheap prefilter — features were never built, so the
            # usable-row count is genuinely unknown. Say that, don't invent it.
            self.detail = (f'{games} games found; needs at least '
                           f'{PREFILTER_GAMES} before a projection is attempted')
        else:
            self.detail = (f'{games} games found; {usable} usable feature rows; '
                           f'{MIN_USABLE_ROWS} required')
        super().__init__(f'insufficient_data: {self.detail}')


def run_prediction(player_id, pitcher_id, is_home, park_team,
                   temp_f, wind_speed, wind_dir, game_date='',
                   fetch_logs=None):
    """
    Project a batter's HRR for one game.

    Raises InsufficientData (a RuntimeError) when there isn't enough history.
    Game View relies on that being an exception rather than a None return:
    st.cache_data does not cache exceptions, so a failed API fetch retries on
    the next page load instead of sticking for 24 hours.

    `fetch_logs` lets a caller supply its own cached game-log fetcher.
    """
    fetch_logs = fetch_logs or _default_fetch_logs

    df = fetch_logs(player_id)
    if df.empty or len(df) < PREFILTER_GAMES:
        raise InsufficientData(len(df))

    # Freeze at pre-game state — exclude game day and later, so a rating never
    # sees the result of the game it is predicting.
    if game_date:
        try:
            cutoff = pd.Timestamp(game_date).date()
            df = df[df['date'].dt.date < cutoff].copy()
        except Exception:
            pass
    if len(df) < PREFILTER_GAMES:
        raise InsufficientData(len(df))

    # fast_mode=False looks up the real starting pitcher for every historical
    # game, so the model sees genuine matchup variance instead of a constant.
    df_feat = build_features(df, fetch_weather=False,
                             override_pitcher_id=pitcher_id, fast_mode=False)
    idx = df_feat.index[-1]
    df_feat.at[idx, 'is_home']     = int(is_home)
    df_feat.at[idx, 'park_factor'] = get_park_factor(park_team)
    df_feat.at[idx, 'temp_f']      = temp_f
    df_feat.at[idx, 'wind_speed']  = wind_speed
    df_feat.at[idx, 'wind_dir']    = wind_dir

    # Pitcher features are real here (fast_mode=False), so they carry signal
    # rather than the zero-variance filler Game View used to train on.
    fc = get_feature_cols(include_pitcher=True)
    dc = df_feat.dropna(subset=fc).reset_index(drop=True)
    if len(dc) < MIN_USABLE_ROWS:
        raise InsufficientData(len(df), usable=len(dc))

    # Train on all rows except the last — the last row is the prediction
    # template. Its rolling features (built with shift(1)) represent stats going
    # into that game, making it a clean out-of-sample prediction row.
    X = dc.iloc[:-1][fc].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = dc.iloc[:-1][TARGET_COL]

    # Recency weights — a game from 3 months ago carries ~37% the weight of
    # today's, so current form dominates without discarding older history.
    _today    = pd.Timestamp('today').normalize()
    _dates    = pd.to_datetime(dc.iloc[:-1]['date'], errors='coerce')
    _days_ago = (_today - _dates).dt.days.fillna(180).clip(lower=0)
    sample_weights = np.exp(-_days_ago / 90).values

    # Power hitters get deeper trees and more estimators to pick up the HR/RBI
    # signal; contact hitters stay shallower and focus on hits and run frequency.
    hr_rate  = (df['hr'].sum() / df['ab'].sum()) if df['ab'].sum() > 0 else 0.0
    is_power = hr_rate >= 0.04

    xgb_params = dict(learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
                      random_state=42, verbosity=0)
    xgb_params.update(n_estimators=120, max_depth=5) if is_power else \
        xgb_params.update(n_estimators=100, max_depth=4)

    xgb = XGBRegressor(**xgb_params)
    xgb.fit(X, y, sample_weight=sample_weights)
    if HAS_LGBM:
        lgb = LGBMRegressor(n_estimators=xgb_params['n_estimators'],
                            learning_rate=0.08, max_depth=xgb_params['max_depth'],
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbose=-1)
        lgb.fit(X, y, sample_weight=sample_weights)

    latest = dc.iloc[-1:].copy()
    latest.at[latest.index[0], 'is_home']     = int(is_home)
    latest.at[latest.index[0], 'park_factor'] = get_park_factor(park_team)
    latest.at[latest.index[0], 'temp_f']      = temp_f
    latest.at[latest.index[0], 'wind_speed']  = wind_speed
    latest.at[latest.index[0], 'wind_dir']    = wind_dir
    if game_date and 'days_since_last_game' in latest.columns:
        try:
            _pred_dt = pd.Timestamp(game_date).date()
            _last_dt = df['date'].max()
            if hasattr(_last_dt, 'date'):
                _last_dt = _last_dt.date()
            latest.at[latest.index[0], 'days_since_last_game'] = float(max(0, (_pred_dt - _last_dt).days))
        except Exception:
            pass

    latest_X = latest[fc].apply(pd.to_numeric, errors='coerce').fillna(0)
    xgb_pred = float(xgb.predict(latest_X)[0])
    if HAS_LGBM:
        proj = max(0.0, xgb_pred * 0.55 + float(lgb.predict(latest_X)[0]) * 0.45)
    else:
        proj = max(0.0, xgb_pred)

    season_avg_val = (float(dc['total_season_avg'].iloc[-1])
                      if not np.isnan(dc['total_season_avg'].iloc[-1]) else 0)
    r30_avg = float((df.tail(30)['h'] + df.tail(30)['r'] + df.tail(30)['rbi']).mean())

    # Floor — a bad feature value shouldn't produce an absurdly low projection
    proj = max(proj, max(season_avg_val * 0.30, r30_avg * 0.30))

    # Ceiling — scales with the player's own averages, no hard cap
    ceiling = max(r30_avg * PROJ_CEILING_MULT, season_avg_val * PROJ_CEILING_MULT, 2.0)
    proj    = min(proj, ceiling)

    if APPLY_PITCHER_MULTIPLIER:
        proj = proj * pitcher_multiplier(dc.iloc[-1])
    proj = round(proj, 2)

    r7  = df.tail(7);  hrr7  = (r7['h']  + r7['r']  + r7['rbi']).mean()
    r30 = df.tail(30); hrr30 = (r30['h'] + r30['r'] + r30['rbi']).mean()
    ab30 = r30['ab'].sum(); h30 = r30['h'].sum(); bb30 = r30['bb'].sum()
    sb30 = r30['sb'].sum() if 'sb' in r30.columns else 0

    hg = df[df['is_home'] == 1]
    ag = df[df['is_home'] == 0]
    home_hrr = (hg['h'] + hg['r'] + hg['rbi']).mean() if len(hg) >= 5 else None
    away_hrr = (ag['h'] + ag['r'] + ag['rbi']).mean() if len(ag) >= 5 else None

    return {
        'proj': round(proj, 2),
        # The post-XGBoost multipliers (pitcher x matchup x context x
        # calibration) stack past this, so callers re-cap the final projection
        # here after calibration rather than trusting the raw stack.
        'proj_ceiling': round(ceiling, 2),
        'r7g':      round(float(hrr7), 2),
        'r30g':     round(float(hrr30), 2),
        'savg':     round(season_avg_val, 2),
        'ba30':     round(h30 / ab30, 3) if ab30 > 0 else 0.250,
        'obp30':    round((h30 + bb30) / (ab30 + bb30), 3) if (ab30 + bb30) > 0 else 0.320,
        'sb_rate':  round(sb30 / 30, 3),
        'home_hrr': round(float(home_hrr), 2) if home_hrr else None,
        'away_hrr': round(float(away_hrr), 2) if away_hrr else None,
        'r20g_venue': round(float(dc['hrr_20g_venue'].iloc[-1]), 2)
                      if 'hrr_20g_venue' in dc.columns and not np.isnan(dc['hrr_20g_venue'].iloc[-1]) else None,
        'ba_venue':   round(float(dc['ba_20g_venue'].iloc[-1]), 3)
                      if 'ba_20g_venue' in dc.columns and not np.isnan(dc['ba_20g_venue'].iloc[-1]) else None,
        'df': df,
    }
