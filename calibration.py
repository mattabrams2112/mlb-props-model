"""
Projection calibration — reads full_play_log, computes mean bias by rating tier,
and returns multipliers to correct systematic over/underconfidence.

Usage:
    from calibration import get_correction_factor
    corrected_proj = raw_proj * get_correction_factor(rating)

Run standalone to print the full calibration report:
    python calibration.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

_CORRECTION_CACHE: dict = {}
_CACHE_TTL = 3600  # re-compute at most once per hour
_LAST_COMPUTED: float = 0.0

TIERS = [
    (90, 101, '90+'),
    (80, 90,  '80-89'),
    (70, 80,  '70-79'),
    (60, 70,  '60-69'),
    (0,  60,  '<60'),
]
MIN_SAMPLE = 20  # minimum decided plays per tier to trust the calibration


def _load_play_log() -> pd.DataFrame:
    """Load full_play_log from DB or CSV."""
    db_url = os.environ.get('DATABASE_URL', '')
    if db_url:
        try:
            from sqlalchemy import create_engine
            url = db_url.replace('postgres://', 'postgresql://', 1)
            if '?' not in url:
                url += '?sslmode=require'
            engine = create_engine(url, connect_args={'connect_timeout': 10})
            return pd.read_sql('SELECT * FROM full_play_log', engine)
        except Exception:
            pass
    path = os.path.join(os.path.dirname(__file__), 'full_play_log.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def compute_calibration() -> dict:
    """
    Returns dict of {tier_label: multiplier}.
    multiplier = mean(actual) / mean(projected) for decided plays in that tier.
    1.0 = perfectly calibrated. >1.0 = model underprojects. <1.0 = model overprojects.
    """
    global _CORRECTION_CACHE, _LAST_COMPUTED
    import time
    now = time.time()
    if _CORRECTION_CACHE and (now - _LAST_COMPUTED) < _CACHE_TTL:
        return _CORRECTION_CACHE

    df = _load_play_log()
    result = {}

    if df.empty or 'projected' not in df.columns or 'actual' not in df.columns:
        _CORRECTION_CACHE = result
        _LAST_COMPUTED = now
        return result

    df['projected'] = pd.to_numeric(df['projected'], errors='coerce')
    df['actual']    = pd.to_numeric(df['actual'],    errors='coerce')
    df['rating']    = pd.to_numeric(df['rating'],    errors='coerce')

    # Only use rows with decided actuals
    decided = df.dropna(subset=['projected', 'actual', 'rating'])
    decided = decided[decided['actual'] > 0]

    for lo, hi, label in TIERS:
        tier = decided[(decided['rating'] >= lo) & (decided['rating'] < hi)]
        if len(tier) < MIN_SAMPLE:
            result[label] = 1.0
            continue
        mean_proj   = tier['projected'].mean()
        mean_actual = tier['actual'].mean()
        if mean_proj > 0:
            result[label] = round(mean_actual / mean_proj, 4)
        else:
            result[label] = 1.0

    _CORRECTION_CACHE = result
    _LAST_COMPUTED = now
    return result


def get_correction_factor(rating: float) -> float:
    """Return the calibration multiplier for a given rating."""
    factors = compute_calibration()
    if not factors:
        return 1.0
    for lo, hi, label in TIERS:
        if lo <= rating < hi:
            return factors.get(label, 1.0)
    return 1.0


if __name__ == '__main__':
    factors = compute_calibration()
    df = _load_play_log()

    if df.empty:
        print('No play log data found.')
        sys.exit(0)

    df['projected'] = pd.to_numeric(df['projected'], errors='coerce')
    df['actual']    = pd.to_numeric(df['actual'],    errors='coerce')
    df['rating']    = pd.to_numeric(df['rating'],    errors='coerce')
    decided = df.dropna(subset=['projected', 'actual', 'rating'])
    decided = decided[decided['actual'] > 0]

    print(f'\n{"=" * 60}')
    print(f'  Projection Calibration Report  ({len(decided)} decided plays)')
    print(f'{"=" * 60}')
    print(f'{"Tier":<10} {"N":>5} {"Avg Proj":>10} {"Avg Actual":>12} {"Multiplier":>12} {"Bias":>10}')
    print(f'{"-" * 60}')

    for lo, hi, label in TIERS:
        tier = decided[(decided['rating'] >= lo) & (decided['rating'] < hi)]
        n = len(tier)
        if n == 0:
            print(f'{label:<10} {"0":>5} {"—":>10} {"—":>12} {"—":>12} {"—":>10}')
            continue
        mp  = tier['projected'].mean()
        ma  = tier['actual'].mean()
        mul = factors.get(label, 1.0)
        mae = (tier['actual'] - tier['projected']).abs().mean()
        bias_str = f'{(ma - mp):+.3f}'
        flag = '  ⚠️  OVERPROJECTS' if mul < 0.90 else '  ⚠️  UNDERPROJECTS' if mul > 1.10 else ''
        print(f'{label:<10} {n:>5} {mp:>10.3f} {ma:>12.3f} {mul:>12.4f} {bias_str:>10}{flag}')

    print(f'{"=" * 60}')
    overall_mae = (decided['actual'] - decided['projected']).abs().mean()
    print(f'Overall MAE: {overall_mae:.3f}')
    print()
