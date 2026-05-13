"""Batter vs Pitcher head-to-head career stats, cached locally."""
import os
import statsapi
import pandas as pd

from data_dir import data_path
CACHE_FILE = data_path('cache_bvp.csv')
BVP_DEFAULT = {'bvp_ab': 0, 'bvp_avg': 0.250, 'bvp_hr': 0, 'bvp_sample': 0}


def _parse_avg(val) -> float:
    try:
        v = str(val).strip().lstrip('.')
        return float('0.' + v) if v and not v.startswith('0') else float(v or '0')
    except (ValueError, TypeError):
        return 0.250


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


def get_bvp(batter_id: int, pitcher_id: int) -> dict:
    cache = _load_cache()
    key = f"{batter_id}_{pitcher_id}"
    if key in cache:
        return cache[key]

    result = BVP_DEFAULT.copy()
    try:
        data = statsapi.get('stats', {
            'personId': batter_id,
            'stats': 'vsPlayer',
            'group': 'hitting',
            'opposingPlayerId': pitcher_id,
        })
        splits = (data.get('stats') or [{}])[0].get('splits', [])
        if splits:
            s = splits[0].get('stat', {})
            ab = int(s.get('atBats', 0))
            h = int(s.get('hits', 0))
            hr = int(s.get('homeRuns', 0))
            result = {
                'bvp_ab':     ab,
                'bvp_avg':    round(h / ab, 3) if ab > 0 else 0.250,
                'bvp_hr':     hr,
                'bvp_sample': 1 if ab >= 10 else 0,
            }
    except Exception:
        pass

    cache[key] = result
    _save_cache(cache)
    return result
