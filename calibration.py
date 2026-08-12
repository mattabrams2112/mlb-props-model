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

# Calibration is computed but NOT applied. `get_correction_factor` returns 1.0
# for every rating while this is False.
#
# Until 2026-08-11 this shipped enabled on top of a zeros-dropping bug (see
# below), which inverted the correction's sign for the tier holding most of the
# population: sub-60 projections were multiplied by 1.4837 — inflated 48% — and
# then re-rated on the inflated number, which is how a marginal play climbed
# into a higher tier. Measured on 15,956 decided rows:
#
#     tier     live factor    correct factor
#     90+         0.7484          0.4605
#     80-89       0.8854          0.6681
#     70-79       0.8517          0.6186
#     60-69       0.9799          0.6957
#     <60         1.4837          0.9757   <- sign inversion, 71% of rows
#
# The bug is fixed below, so the factors this module reports are now honest.
# It stays OFF because the honest factors are the known-bad ones: ~0.46 at 90+
# would collapse the bet band outright. Per CLAUDE.md, calibration may only be
# re-enabled together with a re-derived rating threshold — never alone.
CALIBRATION_ENABLED = False

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

    # Only use rows with decided actuals. A 0 IS a decided actual — the batter
    # went 0-for. Dropping those rows was the sign-inversion bug: it discarded
    # 32.7% of the sample, all of it on the losing side, so mean(actual) rose
    # far above the truth and the multiplier told the model to project higher.
    decided = df.dropna(subset=['projected', 'actual', 'rating'])

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
    """Return the calibration multiplier for a given rating.

    Returns 1.0 unconditionally while CALIBRATION_ENABLED is False. This is the
    single choke point for all three call sites (worker.py, Game View x2), so
    the flag is the whole off switch — no call site needs to know.
    """
    if not CALIBRATION_ENABLED:
        return 1.0
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

    print(f'\n{"=" * 60}')
    status = 'APPLIED LIVE' if CALIBRATION_ENABLED else 'COMPUTED ONLY — NOT APPLIED'
    print(f'  Projection Calibration Report  ({len(decided)} decided plays)')
    print(f'  Status: {status}')
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
