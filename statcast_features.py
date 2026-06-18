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
_MEM_CACHE: dict = {}  # in-memory cache — populated once per process, avoiding repeated CSV reads

PITCH_GROUPS = {
    'fb': ['FF', 'SI', 'FC', 'FT', 'FA'],
    'bk': ['SL', 'CU', 'KC', 'SV', 'ST', 'CS'],
    'os': ['CH', 'FS', 'FO', 'SC', 'KN'],
}

BATTER_DEFAULTS = {
    'batter_fb_barrel_pct': 0.080, 'batter_fb_seen_pct': 0.55,
    'batter_bk_barrel_pct': 0.040, 'batter_bk_seen_pct': 0.25,
    'batter_os_barrel_pct': 0.050, 'batter_os_seen_pct': 0.20,
    'batter_whiff_pct_fb':  0.200, 'batter_whiff_pct_bk': 0.330, 'batter_whiff_pct_os': 0.310,
    'batter_xba':           0.250,
    'batter_xwoba':         0.320,
    'batter_hard_hit_pct':  0.360,
    'batter_avg_ev':        88.0,
    'batter_xba_vs_rhp':    0.250,
    'batter_xba_vs_lhp':    0.250,
    'batter_hard_hit_vs_rhp': 0.360,
    'batter_hard_hit_vs_lhp': 0.360,
    'batter_k_pct':          0.222,
    'batter_bb_pct':         0.083,
    'batter_babip':          0.300,
    'batter_whiff_pct':      0.245,
    'batter_k_pct_vs_rhp':   0.222,
    'batter_k_pct_vs_lhp':   0.222,
    'batter_bb_pct_vs_rhp':  0.083,
    'batter_bb_pct_vs_lhp':  0.083,
    'batter_babip_vs_rhp':   0.300,
    'batter_babip_vs_lhp':   0.300,
}
PITCHER_DEFAULTS = {
    'pitcher_fb_barrel_pct': 0.080, 'pitcher_fb_thrown_pct': 0.55,
    'pitcher_bk_barrel_pct': 0.040, 'pitcher_bk_thrown_pct': 0.25,
    'pitcher_os_barrel_pct': 0.050, 'pitcher_os_thrown_pct': 0.20,
    'pitcher_xba_allowed':   0.250,
    'pitcher_hard_hit_pct':  0.360,
    'pitcher_avg_ev':        88.0,
    'pitcher_gb_pct':        0.430,  # league avg ~43%
    'pitcher_whiff_pct_fb':  0.200, 'pitcher_whiff_pct_bk': 0.330, 'pitcher_whiff_pct_os': 0.310,
    'pitcher_k_pct':          0.222,
    'pitcher_bb_pct':         0.083,
    'pitcher_babip':          0.300,
    'pitcher_whiff_pct':      0.245,
    'pitcher_k_pct_vs_lhb':   0.222,
    'pitcher_k_pct_vs_rhb':   0.222,
    'pitcher_bb_pct_vs_lhb':  0.083,
    'pitcher_bb_pct_vs_rhb':  0.083,
    'pitcher_babip_vs_lhb':   0.300,
    'pitcher_babip_vs_rhb':   0.300,
}


def _season_range(season: int) -> tuple:
    end = min(f'{season}-11-05', datetime.now().strftime('%Y-%m-%d'))
    return f'{season}-03-20', end


def _load_cache() -> dict:
    global _MEM_CACHE
    if _MEM_CACHE:
        return _MEM_CACHE
    if not os.path.exists(CACHE_FILE):
        return _MEM_CACHE
    try:
        df = pd.read_csv(CACHE_FILE, dtype={'key': str})
        if not df.empty and 'key' in df.columns:
            _MEM_CACHE = df.set_index('key').to_dict('index')
    except Exception:
        pass
    return _MEM_CACHE


def _save_cache(cache: dict):
    global _MEM_CACHE
    _MEM_CACHE = cache
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
        # Per-pitch-group whiff% (swinging strikes / pitches of that type)
        if 'description' in group_all.columns and len(group_all) >= 10:
            g_whiff = group_all['description'].isin(
                ['swinging_strike', 'swinging_strike_blocked']).sum()
            result[f'{role}_whiff_pct_{group}'] = round(g_whiff / len(group_all), 4)

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

    # ── Ground ball % (pitcher only) ─────────────────────────────────────────
    if role == 'pitcher' and 'bb_type' in batted.columns:
        n_batted = len(batted)
        if n_batted >= 20:
            gb_count = (batted['bb_type'] == 'ground_ball').sum()
            result['pitcher_gb_pct'] = round(int(gb_count) / n_batted, 4)

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

    # ── K%, BB%, BABIP, Whiff% (both roles) ──────────────────────────────────
    pa_rows = df[df['events'].notna() & (df['events'] != '')] if 'events' in df.columns else pd.DataFrame()
    if not pa_rows.empty:
        pa_count = pa_rows['at_bat_number'].nunique() if 'at_bat_number' in pa_rows.columns else len(pa_rows)
        k_count  = pa_rows['events'].isin(['strikeout', 'strikeout_double_play']).sum()
        bb_count = pa_rows['events'].isin(['walk', 'intent_walk']).sum()
        if pa_count >= 20:
            result[f'{role}_k_pct']  = round(k_count  / pa_count, 3)
            result[f'{role}_bb_pct'] = round(bb_count / pa_count, 3)

    # BABIP: hits on balls in play / (balls in play - HR)
    if len(batted) >= 20 and 'events' in batted.columns:
        h_bip  = batted['events'].isin(['single', 'double', 'triple']).sum()
        hr_ct  = batted['events'].isin(['home_run']).sum()
        bip    = len(batted) - hr_ct
        if bip > 0:
            result[f'{role}_babip'] = round(h_bip / bip, 3)

    # Whiff%: swinging strikes / total pitches
    if 'description' in df.columns and len(df) >= 50:
        whiff = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).sum()
        result[f'{role}_whiff_pct'] = round(whiff / len(df), 3)

    # Platoon K%, BB%, BABIP (batter side: split by pitcher hand; pitcher side: split by batter hand)
    split_col   = 'p_throws' if role == 'batter' else 'stand'
    split_sides = [('R', 'rhp'), ('L', 'lhp')] if role == 'batter' else [('L', 'lhb'), ('R', 'rhb')]
    if split_col in df.columns:
        for hand, suffix in split_sides:
            s_df     = df[df[split_col] == hand]
            s_batted = s_df[s_df['type'] == 'X'] if 'type' in s_df.columns else pd.DataFrame()
            # K%, BB%
            if 'events' in s_df.columns:
                s_pa = s_df[s_df['events'].notna() & (s_df['events'] != '')]
                s_pa_count = s_pa['at_bat_number'].nunique() if 'at_bat_number' in s_pa.columns else len(s_pa)
                if s_pa_count >= 10:
                    s_k  = s_pa['events'].isin(['strikeout', 'strikeout_double_play']).sum()
                    s_bb = s_pa['events'].isin(['walk', 'intent_walk']).sum()
                    result[f'{role}_k_pct_vs_{suffix}']  = round(s_k  / s_pa_count, 3)
                    result[f'{role}_bb_pct_vs_{suffix}'] = round(s_bb / s_pa_count, 3)
            # BABIP
            if len(s_batted) >= 10 and 'events' in s_batted.columns:
                s_h   = s_batted['events'].isin(['single', 'double', 'triple']).sum()
                s_hr  = s_batted['events'].isin(['home_run']).sum()
                s_bip = len(s_batted) - s_hr
                if s_bip > 0:
                    result[f'{role}_babip_vs_{suffix}'] = round(s_h / s_bip, 3)

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
    if key in cache and 'batter_k_pct' in cache[key]:
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
    if key in cache and 'pitcher_k_pct' in cache[key]:
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
