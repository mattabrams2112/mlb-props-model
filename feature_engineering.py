import pandas as pd
import numpy as np
from weather import fetch_weather_for_games, get_park_factor
from pitcher_data import (get_starting_pitchers_for_games, get_pitcher_season_stats,
                          get_rolling_pitcher_stats, LEAGUE_AVG)
from bvp_stats import get_bvp
from statcast_features import (
    get_batter_statcast, get_pitcher_statcast,
    BATTER_STATCAST_COLS, PITCHER_STATCAST_COLS,
    BATTER_DEFAULTS, PITCHER_DEFAULTS,
)

WINDOWS = [7, 14, 20, 30]
STAT_COLS = ['h', 'r', 'rbi', 'hr', 'bb', 'k', 'ab', 'd', 't', 'total']
TARGET_COL = 'total'

PITCHER_FEATURE_COLS = ['opp_era', 'opp_whip', 'opp_k_pct', 'opp_bb_pct', 'opp_h_per_9']
BVP_FEATURE_COLS = ['bvp_avg', 'bvp_ab', 'bvp_sample']


def _add_pitcher_features(df: pd.DataFrame, override_pitcher_id: int = None,
                          fast_mode: bool = False) -> pd.DataFrame:
    """
    fast_mode=False (training): looks up starting pitcher per game from boxscores,
    then uses rolling stats (last 5 starts) so the model sees real matchup signal.
    fast_mode=True (live/Game View): fills history with league averages for speed,
    applies override_pitcher stats to the prediction row only.
    """
    if 'game_pk' not in df.columns or fast_mode:
        # Fill all rows with league averages
        for col in PITCHER_FEATURE_COLS:
            df[col] = LEAGUE_AVG.get(col, 0.0)
        for col in BVP_FEATURE_COLS:
            df[col] = 0.0
        for col in PITCHER_STATCAST_COLS:
            df[col] = PITCHER_DEFAULTS.get(col, 0.0)
        # Still apply override pitcher to most recent row
        if override_pitcher_id is not None:
            batter_id = int(df['player_id'].iloc[0]) if 'player_id' in df.columns else None
            season    = int(df['season'].iloc[-1])
            ov_stats  = get_pitcher_season_stats(override_pitcher_id, season)
            ov_sc     = get_pitcher_statcast(override_pitcher_id, season)
            ov_bvp    = get_bvp(batter_id, override_pitcher_id) if batter_id else {}
            idx = df.index[-1]
            for col in PITCHER_FEATURE_COLS:
                df.at[idx, col] = ov_stats.get(col, LEAGUE_AVG.get(col, 0.0))
            for col in PITCHER_STATCAST_COLS:
                df.at[idx, col] = ov_sc.get(col, PITCHER_DEFAULTS[col])
            for col in BVP_FEATURE_COLS:
                df.at[idx, col] = ov_bvp.get(col, 0.0)
        return df

    valid_pks = df['game_pk'].dropna()
    valid_pks = valid_pks[valid_pks != ''].tolist()
    game_pitcher_map = get_starting_pitchers_for_games(valid_pks)

    pitcher_stats_rows = []
    pitcher_sc_rows = []
    bvp_rows = []
    batter_id = int(df['player_id'].iloc[0]) if 'player_id' in df.columns else None

    for _, row in df.iterrows():
        pk = str(row.get('game_pk', ''))
        season = int(row.get('season', 0))
        is_home = int(row.get('is_home', 1))
        game_date = row.get('date')

        game_pitchers = game_pitcher_map.get(pk, {})
        pitcher_id = (
            game_pitchers.get('away_pitcher_id') if is_home
            else game_pitchers.get('home_pitcher_id')
        )

        if pitcher_id:
            pid = int(pitcher_id)
            pitcher_stats_rows.append(
                get_rolling_pitcher_stats(pid, game_date, season, batter_is_home=is_home)
            )
            pitcher_sc_rows.append(get_pitcher_statcast(pid, season))
        else:
            pitcher_stats_rows.append(LEAGUE_AVG.copy())
            pitcher_sc_rows.append(PITCHER_DEFAULTS.copy())

        bvp_rows.append(
            get_bvp(batter_id, int(pitcher_id))
            if batter_id and pitcher_id
            else {'bvp_avg': 0.0, 'bvp_ab': 0, 'bvp_sample': 0}
        )

    for col in PITCHER_FEATURE_COLS:
        df[col] = [r.get(col, LEAGUE_AVG.get(col, 0.0)) for r in pitcher_stats_rows]
    for col in BVP_FEATURE_COLS:
        df[col] = [r.get(col, 0.0) for r in bvp_rows]
    for col in PITCHER_STATCAST_COLS:
        df[col] = [r.get(col, PITCHER_DEFAULTS[col]) for r in pitcher_sc_rows]

    # Override everything for the most recent row when a specific pitcher is supplied
    if override_pitcher_id is not None:
        season = int(df['season'].iloc[-1])
        ov_stats = get_pitcher_season_stats(override_pitcher_id, season)
        ov_sc    = get_pitcher_statcast(override_pitcher_id, season)
        ov_bvp   = get_bvp(batter_id, override_pitcher_id) if batter_id else {}
        idx = df.index[-1]
        for col in PITCHER_FEATURE_COLS:
            df.at[idx, col] = ov_stats.get(col, LEAGUE_AVG.get(col, 0.0))
        for col in PITCHER_STATCAST_COLS:
            df.at[idx, col] = ov_sc.get(col, PITCHER_DEFAULTS[col])
        for col in BVP_FEATURE_COLS:
            df.at[idx, col] = ov_bvp.get(col, 0.0)

    return df


def _add_batter_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """Add batter Statcast barrel rates and pitch-mix seen pcts, per season."""
    if 'player_id' not in df.columns:
        for col in BATTER_STATCAST_COLS:
            df[col] = BATTER_DEFAULTS[col]
        return df

    batter_id = int(df['player_id'].iloc[0])

    # Fetch once per unique season in the dataset
    season_cache = {}
    for season in df['season'].unique():
        season_cache[int(season)] = get_batter_statcast(batter_id, int(season))

    for col in BATTER_STATCAST_COLS:
        df[col] = df['season'].apply(lambda s: season_cache.get(int(s), BATTER_DEFAULTS)[col])

    return df


def build_features(df: pd.DataFrame, fetch_weather: bool = True,
                   override_pitcher_id: int = None, fast_mode: bool = False) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    df['total'] = df['h'] + df['r'] + df['rbi']

    for col in STAT_COLS:
        for w in WINDOWS:
            df[f'{col}_avg_{w}g'] = df[col].shift(1).rolling(w, min_periods=max(5, w // 2)).mean()

    for w in WINDOWS:
        hits = df['h'].shift(1).rolling(w, min_periods=max(5, w // 2)).sum()
        ab   = df['ab'].shift(1).rolling(w, min_periods=max(5, w // 2)).sum()
        tb   = (df['h'] + df['d'] + 2 * df['t'] + 3 * df['hr']).shift(1).rolling(w, min_periods=max(5, w // 2)).sum()
        df[f'ba_{w}g']  = hits / ab.replace(0, np.nan)
        df[f'slg_{w}g'] = tb   / ab.replace(0, np.nan)

    df['total_season_avg'] = (
        df.groupby('season')['total']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # ── Home/away rolling hit rates (last 20 games in each venue) ─────────────
    # These capture venue-specific recent form — more predictive than overall avg
    for venue_val, suffix in [(1, 'home'), (0, 'away')]:
        mask = df['is_home'] == venue_val
        # Rolling 20g HRR in this venue (no leakage)
        venue_total = df['total'].where(mask).fillna(np.nan)
        df[f'hrr_20g_{suffix}'] = (
            venue_total.shift(1).rolling(20, min_periods=5).mean()
        )
        # Rolling 20g BA in this venue
        venue_h  = df['h'].where(mask).fillna(np.nan)
        venue_ab = df['ab'].where(mask).fillna(np.nan)
        df[f'ba_20g_{suffix}'] = (
            venue_h.shift(1).rolling(20, min_periods=5).sum() /
            venue_ab.shift(1).rolling(20, min_periods=5).sum().replace(0, np.nan)
        )

    # Fill missing venue splits with overall averages
    df['hrr_20g_home'] = df['hrr_20g_home'].fillna(df['total_avg_20g'])
    df['hrr_20g_away'] = df['hrr_20g_away'].fillna(df['total_avg_20g'])
    df['ba_20g_home']  = df['ba_20g_home'].fillna(df['ba_20g'])
    df['ba_20g_away']  = df['ba_20g_away'].fillna(df['ba_20g'])

    # Current venue HRR and BA (whichever applies to this game)
    df['hrr_20g_venue'] = np.where(df['is_home'] == 1, df['hrr_20g_home'], df['hrr_20g_away'])
    df['ba_20g_venue']  = np.where(df['is_home'] == 1, df['ba_20g_home'],  df['ba_20g_away'])

    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['park_factor'] = df['home_team'].apply(get_park_factor)

    # Day/night game (UTC hour: < 17 = day game ~1pm ET, >= 17 = night ~7pm ET)
    if 'game_hour' in df.columns:
        df['is_day_game'] = (df['game_hour'] < 17).astype(int)
    else:
        df['is_day_game'] = 0

    # Rolling K% and BB% for batter
    for w in WINDOWS:
        pa = (df['ab'] + df['bb']).shift(1).rolling(w, min_periods=3).sum()
        df[f'k_pct_{w}g']  = df['k'].shift(1).rolling(w, min_periods=3).sum() / pa.replace(0, np.nan)
        df[f'bb_pct_{w}g'] = df['bb'].shift(1).rolling(w, min_periods=3).sum() / pa.replace(0, np.nan)

    # Rolling BABIP = (H - HR) / (AB - K - HR)
    for w in WINDOWS:
        h_r  = df['h'].shift(1).rolling(w, min_periods=3).sum()
        hr_r = df['hr'].shift(1).rolling(w, min_periods=3).sum()
        ab_r = df['ab'].shift(1).rolling(w, min_periods=3).sum()
        k_r  = df['k'].shift(1).rolling(w, min_periods=3).sum()
        denom = (ab_r - k_r - hr_r).replace(0, np.nan)
        df[f'babip_{w}g'] = (h_r - hr_r) / denom

    # Home/away splits — rolling avg HRR at home vs away (no leakage)
    try:
        df['home_hrr_avg'] = (
            df.groupby(['season', 'is_home'])['total']
            .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
        )
        # Fall back to season avg, then 30g avg, then overall mean
        overall_mean = df['total'].mean()
        df['home_hrr_avg'] = (df['home_hrr_avg']
                              .fillna(df['total_season_avg'])
                              .fillna(df['total'].shift(1).rolling(30, min_periods=3).mean())
                              .fillna(overall_mean)
                              .fillna(0.0))
    except Exception:
        df['home_hrr_avg'] = df['total'].shift(1).rolling(30, min_periods=3).mean().fillna(0.0)

    # Weather
    if fetch_weather and 'game_pk' in df.columns:
        valid_pks = df['game_pk'].dropna()
        valid_pks = valid_pks[valid_pks != ''].tolist()
        if valid_pks:
            weather_df = fetch_weather_for_games(valid_pks)
            df = df.merge(weather_df[['game_pk', 'temp_f', 'wind_speed', 'wind_dir']],
                          on='game_pk', how='left')
        else:
            df['temp_f'] = None; df['wind_speed'] = 0.0; df['wind_dir'] = 0
    else:
        df['temp_f'] = None; df['wind_speed'] = 0.0; df['wind_dir'] = 0

    df['temp_f']     = df['temp_f'].fillna(72.0)
    df['wind_speed'] = df['wind_speed'].fillna(0.0)
    df['wind_dir']   = df['wind_dir'].fillna(0)

    # Batter Statcast (barrel rates + pitch-mix seen)
    df = _add_batter_statcast(df)

    # Pitcher season stats + Statcast (barrel rates + pitch-mix thrown) + BvP
    df = _add_pitcher_features(df, override_pitcher_id=override_pitcher_id, fast_mode=fast_mode)

    # Quality-adjusted rolling HRR — games vs tough pitchers weighted more
    # Uses opp_era per row (already filled above); lower ERA = tougher opponent = higher weight
    LEAGUE_ERA = 4.30
    if 'opp_era' in df.columns:
        opp_era_safe = df['opp_era'].replace(0, LEAGUE_ERA).fillna(LEAGUE_ERA)
        quality_weight = LEAGUE_ERA / opp_era_safe  # >1 for tough pitchers, <1 for easy ones
        weighted_total = df['total'] * quality_weight
        for w in [7, 14, 20]:
            df[f'qa_hrr_{w}g'] = (
                weighted_total.shift(1).rolling(w, min_periods=max(3, w // 2)).mean()
            )
        df['qa_hrr_7g']  = df['qa_hrr_7g'].fillna(df['total_avg_7g'])
        df['qa_hrr_14g'] = df['qa_hrr_14g'].fillna(df['total_avg_14g'])
        df['qa_hrr_20g'] = df['qa_hrr_20g'].fillna(df['total_avg_20g'])
    else:
        for w in [7, 14, 20]:
            df[f'qa_hrr_{w}g'] = df[f'total_avg_{w}g']

    return df


def get_feature_cols(include_pitcher: bool = True) -> list:
    # Only roll stats that directly predict H+R+RBI — cuts 24 redundant features
    # (hr, bb, k, ab, d, t rolling avgs already captured inside total_avg)
    _rolling = ['h', 'r', 'rbi', 'total']
    cols = []
    for col in _rolling:
        for w in WINDOWS:
            cols.append(f'{col}_avg_{w}g')
    for w in WINDOWS:
        cols += [f'ba_{w}g', f'slg_{w}g']
    cols.append('total_season_avg')
    # Drop day_of_week, is_day_game, wind_speed, wind_dir, temp_f, home_hrr_avg — low signal
    cols += ['is_home', 'month', 'park_factor']
    for w in WINDOWS:
        cols += [f'k_pct_{w}g', f'bb_pct_{w}g', f'babip_{w}g']
    # Drop hrr_20g_home/away and ba_20g_home/away — hrr_20g_venue already picks the right one
    cols += ['hrr_20g_venue', 'ba_20g_venue']
    # Quality-adjusted HRR — performance weighted by opposing pitcher ERA
    cols += ['qa_hrr_7g', 'qa_hrr_14g', 'qa_hrr_20g']
    # Batter Statcast cols (barrel rates, pitch-mix, whiff) are season-level constants —
    # zero game-to-game variance → XGBoost assigns 0 importance. They're already used
    # in the rating engine (Barrel Edge, Contact Quality, Platoon). Exclude from XGBoost.
    if include_pitcher:
        cols += PITCHER_FEATURE_COLS + BVP_FEATURE_COLS + PITCHER_STATCAST_COLS
    return cols
