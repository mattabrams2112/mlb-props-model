"""Batter vs Pitcher head-to-head career stats, cached locally."""
import os
import statsapi
import pandas as pd

import atomic_cache

from data_dir import data_path
CACHE_FILE = data_path('cache_bvp.csv')
BVP_DEFAULT = {'bvp_ab': 0, 'bvp_avg': 0.250, 'bvp_hr': 0, 'bvp_sample': 0}
_MEM_CACHE: dict = {}


def _parse_avg(val) -> float:
    try:
        v = str(val).strip().lstrip('.')
        return float('0.' + v) if v and not v.startswith('0') else float(v or '0')
    except (ValueError, TypeError):
        return 0.250


def _load_cache() -> dict:
    """Returns the shared in-memory cache.

    Still returned by reference, so lookups stay cheap — but every mutation
    must go through `_put`, which holds this file's lock across the mutation,
    the snapshot and the atomic write. Mutating the returned dict directly
    reopens the race this replaced. See atomic_cache for the full account.
    """
    global _MEM_CACHE
    with atomic_cache.lock_for(CACHE_FILE):
        if _MEM_CACHE:
            return _MEM_CACHE
        try:
            loaded = atomic_cache.load_cache(CACHE_FILE, 'key')
        except atomic_cache.CacheCorruptionError as e:
            # A corrupt file is NOT a cold start. Say so, and keep whatever is
            # already in memory rather than silently discarding every entry.
            print(atomic_cache.describe_failure(
                'cache-read-fail', 'bvp_stats', CACHE_FILE, e, 'refetch'))
            return _MEM_CACHE
        if loaded:
            _MEM_CACHE = loaded
        return _MEM_CACHE


def _put(key, value) -> None:
    """Mutate + persist under one lock. The only supported way to write."""
    global _MEM_CACHE
    with atomic_cache.lock_for(CACHE_FILE):
        _MEM_CACHE[key] = value
        # Best-effort persist: the value is in memory and is returned either
        # way, so a disk failure must not become a dropped batter.
        atomic_cache.save_cache_best_effort(CACHE_FILE, 'key', _MEM_CACHE,
                                            module='bvp_stats')


def _save_cache(cache: dict):
    """Back-compat shim: replace the whole cache, atomically."""
    global _MEM_CACHE
    with atomic_cache.lock_for(CACHE_FILE):
        _MEM_CACHE = cache
        atomic_cache.save_cache_best_effort(CACHE_FILE, 'key', _MEM_CACHE,
                                            module='bvp_stats')


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

    _put(key, result)
    return result
