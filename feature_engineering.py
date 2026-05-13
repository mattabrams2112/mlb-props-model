import pandas as pd
import numpy as np
from weather import fetch_weather_for_games, get_park_factor
from pitcher_data import get_starting_pitchers_for_games, get_pitcher_season_stats, LEAGUE_AVG
from bvp_stats import get_bvp
from statcast_features import (
    get_batter_statcast, get_pitcher_statcast,
    BATTER_STATCAST_COLS, PITCHER_STATCAST_COLS,
    BATTER_DEFAULTS, PITCHER_DEFAULTS,
)

WINDOWS = [7, 14, 30]
STAT_COLS = ['h', 'r', 'rbi', 'hr', 'bb', 'k', 'ab', 'd', 't', 'total']
TARGET_COL = 'total'

PITCHER_FEATURE_COLS = ['opp_era', 'opp_whip', 'opp_k_pct', 'opp_bb_pct', 'opp_h_per_9']
BVP_FEATURE_COLS = ['bvp_avg', 'bvp_ab', 'bvp_sample']


def _add_pitcher_features(df: pd.DataFrame, override_pitcher_id: int = None,
                          fast_mode: bool = False) -> pd.DataFrame:
    """
    fast_mode=True skips the 300+ API calls to look up historical starting pitchers.
    Uses league averages for all training rows, only fetching the override pitcher.
    Use this for the Game View page where speed matters.
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

        game_pitchers = game_pitcher_map.get(pk, {})
        pitcher_id = (
            game_pitchers.get('away_pitcher_id') if is_home
            else game_pitchers.get('home_pitcher_id')
        )

        if pitcher_id:
            pid = int(pitcher_id)
            pitcher_stats_rows.append(get_pitcher_season_stats(pid, season))
            pitcher_sc_rows.append(get_pitcher_statcast(pid, season))
        else:
            pitcher_stats_rows.append(LEAGUE_AVG.copy())
            pitcher_sc_rows.append(PITCHER_DEFAULTS.copy())

        bvp_rows.append(
            get_bvp(batter_id, int(pitcher_id))
            if batter_id and pitcher_id
            else {'bvp_avg': 0.250, 'bvp_ab': 0, 'bvp_sample': 0}
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
            df[f'{col}_avg_{w}g'] = df[col].shift(1).rolling(w, min_periods=max(3, w // 3)).mean()

    for w in WINDOWS:
        hits = df['h'].shift(1).rolling(w, min_periods=3).sum()
        ab   = df['ab'].shift(1).rolling(w, min_periods=3).sum()
        tb   = (df['h'] + df['d'] + 2 * df['t'] + 3 * df['hr']).shift(1).rolling(w, min_periods=3).sum()
        df[f'ba_{w}g']  = hits / ab.replace(0, np.nan)
        df[f'slg_{w}g'] = tb   / ab.replace(0, np.nan)

    df['total_season_avg'] = (
        df.groupby('season')['total']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['park_factor'] = df['home_team'].apply(get_park_factor)

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

    return df


def get_feature_cols() -> list:
    cols = []
    for col in STAT_COLS:
        for w in WINDOWS:
            cols.append(f'{col}_avg_{w}g')
    for w in WINDOWS:
        cols += [f'ba_{w}g', f'slg_{w}g']
    cols.append('total_season_avg')
    cols += ['is_home', 'month', 'day_of_week', 'park_factor', 'temp_f', 'wind_speed', 'wind_dir', 'home_hrr_avg']
    cols += PITCHER_FEATURE_COLS + BVP_FEATURE_COLS
    cols += BATTER_STATCAST_COLS + PITCHER_STATCAST_COLS
    return cols
