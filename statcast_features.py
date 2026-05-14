"""
Statcast barrel rates and pitch-mix percentages for batters and pitchers.

Batter features (per pitch group):
  batter_{fb/bk/os}_barrel_pct  — barrel rate on that pitch type
  batter_{fb/bk/os}_seen_pct    — share of pitches seen of that type

Pitcher features (per pitch group):
  pitcher_{fb/bk/os}_barrel_pct — barrel rate allowed on that pitch type
  pitcher_{fb/bk/os}_thrown_pct — share of pitches thrown of that type

Pitch groupings:
  FB (fastball)    : FF, SI, FC, FT, FA
  BK (breaking)   : SL, CU, KC, SV, ST, CS
  OS (offspeed)   : CH, FS, FO, SC, KN
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pybaseball import statcast_batter as _sc_batter, statcast_pitcher as _sc_pitcher

CURRENT_YEAR = datetime.now().year
from data_dir import data_path
CACHE_FILE = data_path('cache_statcast.csv')

PITCH_GROUPS = {
    'fb': ['FF', 'SI', 'FC', 'FT', 'FA'],
    'bk': ['SL', 'CU', 'KC', 'SV', 'ST', 'CS'],
    'os': ['CH', 'FS', 'FO', 'SC', 'KN'],
}

BATTER_DEFAULTS = {
    'batter_fb_barrel_pct': 0.080, 'batter_fb_seen_pct': 0.55,
    'batter_bk_barrel_pct': 0.040, 'batter_bk_seen_pct': 0.25,
    'batter_os_barrel_pct': 0.050, 'batter_os_seen_pct': 0.20,
    'batter_xba':           0.250,
    'batter_xwoba':         0.320,
    'batter_hard_hit_pct':  0.360,
    'batter_avg_ev':        88.0,
    'batter_xba_vs_rhp':    0.250,
    'batter_xba_vs_lhp':    0.250,
    'batter_hard_hit_vs_rhp': 0.360,
    'batter_hard_hit_vs_lhp': 0.360,
}
PITCHER_DEFAULTS = {
    'pitcher_fb_barrel_pct': 0.080, 'pitcher_fb_thrown_pct': 0.55,
    'pitcher_bk_barrel_pct': 0.040, 'pitcher_bk_thrown_pct': 0.25,
    'pitcher_os_barrel_pct': 0.050, 'pitcher_os_thrown_pct': 0.20,
    'pitcher_xba_allowed':   0.250,
    'pitcher_hard_hit_pct':  0.360,
    'pitcher_avg_ev':        88.0,
}


def _season_range(season: int) -> tuple:
    end = min(f'{season}-11-05', datetime.now().strftime('%Y-%m-%d'))
    return f'{season}-03-20', end


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        df = pd.read_csv(CACHE_FILE, dtype={'key': str})
        if df.empty or 'key' not in df.columns:
            return {}
        return df.set_index('key').to_dict('index')
    except Exception:
        return {}


def _save_cache(cache: dict):
    pd.DataFrame([{'key': k, **v} for k, v in cache.items()]).to_csv(CACHE_FILE, index=False)


def _compute_features(df: pd.DataFrame, role: str) -> dict:
    """
    role = 'batter' or 'pitcher'
    Returns barrel rates and pitch-mix pcts for each pitch group.
    """
    if df is None or df.empty:
        return BATTER_DEFAULTS.copy() if role == 'batter' else PITCHER_DEFAULTS.copy()

    df = df[df['pitch_type'].notna()].copy()
    total_pitches = len(df)
    if total_pitches == 0:
        return BATTER_DEFAULTS.copy() if role == 'batter' else PITCHER_DEFAULTS.copy()

    # Batted-ball events only (for barrel rate)
    batted = df[df['type'] == 'X']

    result = {}
    mix_label = 'seen_pct' if role == 'batter' else 'thrown_pct'
    defaults  = BATTER_DEFAULTS if role == 'batter' else PITCHER_DEFAULTS

    for group, pitch_types in PITCH_GROUPS.items():
        prefix       = f'{role}_{group}'
        group_all    = df[df['pitch_type'].isin(pitch_types)]
        group_batted = batted[batted['pitch_type'].isin(pitch_types)]
        result[f'{prefix}_{mix_label}'] = round(len(group_all) / total_pitches, 4)
        n_batted = len(group_batted)
        if n_batted >= 10:
            barrels = (group_batted['launch_speed_angle'] == 6).sum()
            result[f'{prefix}_barrel_pct'] = round(int(barrels) / n_batted, 4)
        else:
            result[f'{prefix}_barrel_pct'] = defaults[f'{prefix}_barrel_pct']

    # ── Advanced Statcast metrics ─────────────────────────────────────────────
    if len(batted) >= 10:
        # xBA / xwOBA — expected stats from exit velo + launch angle
        if 'estimated_ba_using_speedangle' in batted.columns:
            xba = batted['estimated_ba_using_speedangle'].dropna()
            if len(xba) >= 5:
                result[f'{role}_xba' if role == 'batter' else 'pitcher_xba_allowed'] = round(float(xba.mean()), 4)

        if role == 'batter' and 'estimated_woba_using_speedangle' in batted.columns:
            xwoba = batted['estimated_woba_using_speedangle'].dropna()
            if len(xwoba) >= 5:
                result['batter_xwoba'] = round(float(xwoba.mean()), 4)

        # Hard hit rate — exit velo >= 95 mph
        if 'launch_speed' in batted.columns:
            ev = batted['launch_speed'].dropna()
            if len(ev) >= 10:
                hard_hit = (ev >= 95).sum() / len(ev)
                avg_ev   = float(ev.mean())
                result[f'{role}_hard_hit_pct'] = round(hard_hit, 4)
                result[f'{role}_avg_ev']        = round(avg_ev, 2)

    # ── Platoon splits (batter only) ─────────────────────────────────────────
    if role == 'batter' and 'p_throws' in df.columns:
        for side, suffix in [('R', 'rhp'), ('L', 'lhp')]:
            side_df     = df[df['p_throws'] == side]
            side_batted = side_df[side_df['type'] == 'X']
            if len(side_batted) >= 10:
                if 'estimated_ba_using_speedangle' in side_batted.columns:
                    xba_s = side_batted['estimated_ba_using_speedangle'].dropna()
                    if len(xba_s) >= 5:
                        result[f'batter_xba_vs_{suffix}'] = round(float(xba_s.mean()), 4)
                if 'launch_speed' in side_batted.columns:
                    ev_s = side_batted['launch_speed'].dropna()
                    if len(ev_s) >= 5:
                        result[f'batter_hard_hit_vs_{suffix}'] = round((ev_s >= 95).sum() / len(ev_s), 4)

    # Fill any missing advanced metrics with defaults
    for k, v in defaults.items():
        if k not in result:
            result[k] = v

    return result


def get_batter_statcast(player_id: int, season: int = None) -> dict:
    if season is None:
        season = CURRENT_YEAR
    cache = _load_cache()
    key = f'bat_{player_id}_{season}'
    if key in cache:
        return cache[key]

    try:
        start, end = _season_range(season)
        df = _sc_batter(start, end, player_id)
        result = _compute_features(df, 'batter')
    except Exception as e:
        print(f"  Warning: Statcast batter fetch failed for {player_id}: {e}")
        result = BATTER_DEFAULTS.copy()

    cache[key] = result
    _save_cache(cache)
    return result


def get_pitcher_statcast(pitcher_id: int, season: int = None) -> dict:
    if season is None:
        season = CURRENT_YEAR
    cache = _load_cache()
    key = f'pit_{pitcher_id}_{season}'
    if key in cache:
        return cache[key]

    try:
        start, end = _season_range(season)
        df = _sc_pitcher(start, end, pitcher_id)
        result = _compute_features(df, 'pitcher')
    except Exception as e:
        print(f"  Warning: Statcast pitcher fetch failed for {pitcher_id}: {e}")
        result = PITCHER_DEFAULTS.copy()

    cache[key] = result
    _save_cache(cache)
    return result


# Flat list of all column names added by this module
BATTER_STATCAST_COLS = list(BATTER_DEFAULTS.keys())
PITCHER_STATCAST_COLS = list(PITCHER_DEFAULTS.keys())
