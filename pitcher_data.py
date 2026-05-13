"""
Handles two things:
  1. Pitcher season stats (ERA, WHIP, K%, BB%) — cached by pitcher_id + season
  2. Starting pitcher ID per game — cached by game_pk
"""
import os
import time
import statsapi
import pandas as pd
from datetime import datetime

CURRENT_YEAR = datetime.now().year
PITCHER_STATS_CACHE = 'cache_pitcher_stats.csv'
GAME_PITCHER_CACHE = 'cache_game_pitchers.csv'

LEAGUE_AVG = {
    'opp_era': 4.30,
    'opp_whip': 1.28,
    'opp_k_pct': 0.222,
    'opp_bb_pct': 0.083,
    'opp_h_per_9': 8.8,
}


def _parse_float(val, default: float) -> float:
    try:
        v = str(val).strip()
        return float(v) if v and v not in ('---', '.---', 'None', '') else default
    except (ValueError, TypeError):
        return default


# ── Pitcher season stats ──────────────────────────────────────────────────────

def _load_pitcher_cache() -> dict:
    if not os.path.exists(PITCHER_STATS_CACHE):
        return {}
    try:
        df = pd.read_csv(PITCHER_STATS_CACHE, dtype={'key': str})
        if df.empty or 'key' not in df.columns:
            return {}
        return df.set_index('key').to_dict('index')
    except Exception:
        return {}


def _save_pitcher_cache(cache: dict):
    pd.DataFrame([{'key': k, **v} for k, v in cache.items()]).to_csv(PITCHER_STATS_CACHE, index=False)


def get_pitcher_season_stats(pitcher_id: int, season: int = None) -> dict:
    if season is None:
        season = CURRENT_YEAR
    cache = _load_pitcher_cache()
    key = f"{pitcher_id}_{season}"
    if key in cache:
        return cache[key]

    result = LEAGUE_AVG.copy()
    try:
        data = statsapi.player_stat_data(pitcher_id, group='pitching', type='season', season=season)
        for split in data.get('stats', []):
            s = split.get('stat', {})
            ip = _parse_float(s.get('inningsPitched'), 0)
            bb = _parse_float(s.get('baseOnBalls'), 0)
            k = _parse_float(s.get('strikeOuts'), 0)
            h = _parse_float(s.get('hits'), 0)
            bf = _parse_float(s.get('battersFaced'), 1)

            result = {
                'opp_era':    _parse_float(s.get('era'),  LEAGUE_AVG['opp_era']),
                'opp_whip':   _parse_float(s.get('whip'), LEAGUE_AVG['opp_whip']),
                'opp_k_pct':  round(k / bf, 3) if bf > 0 else LEAGUE_AVG['opp_k_pct'],
                'opp_bb_pct': round(bb / bf, 3) if bf > 0 else LEAGUE_AVG['opp_bb_pct'],
                'opp_h_per_9': round(h * 9 / ip, 2) if ip > 0 else LEAGUE_AVG['opp_h_per_9'],
            }
            break
    except Exception:
        pass

    cache[key] = result
    _save_pitcher_cache(cache)
    return result


def get_pitcher_name(pitcher_id: int) -> str:
    try:
        results = statsapi.lookup_player(pitcher_id)
        if results:
            return results[0].get('fullName', str(pitcher_id))
    except Exception:
        pass
    return str(pitcher_id)


# ── Starting pitcher per game ─────────────────────────────────────────────────

def _load_game_pitcher_cache() -> dict:
    if not os.path.exists(GAME_PITCHER_CACHE):
        return {}
    try:
        df = pd.read_csv(GAME_PITCHER_CACHE, dtype={'game_pk': str})
        if df.empty or 'game_pk' not in df.columns:
            return {}
        return df.set_index('game_pk').to_dict('index')
    except Exception:
        return {}


def _save_game_pitcher_cache(cache: dict):
    pd.DataFrame([{'game_pk': k, **v} for k, v in cache.items()]).to_csv(GAME_PITCHER_CACHE, index=False)


def get_starting_pitchers_for_games(game_pks: list, verbose: bool = True) -> dict:
    """Returns {game_pk: {'home_pitcher_id': int|None, 'away_pitcher_id': int|None}}"""
    cache = _load_game_pitcher_cache()
    missing = [str(pk) for pk in game_pks if str(pk) not in cache]

    if missing:
        if verbose:
            print(f"  Fetching starting pitchers for {len(missing)} games "
                  f"({len(game_pks) - len(missing)} cached)...")
        for i, pk in enumerate(missing):
            try:
                box = statsapi.boxscore_data(int(pk))
                home_pitchers = box.get('home', {}).get('pitchers', [])
                away_pitchers = box.get('away', {}).get('pitchers', [])
                cache[pk] = {
                    'home_pitcher_id': home_pitchers[0] if home_pitchers else None,
                    'away_pitcher_id': away_pitchers[0] if away_pitchers else None,
                }
            except Exception:
                cache[pk] = {'home_pitcher_id': None, 'away_pitcher_id': None}

            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(missing)} fetched...")
            time.sleep(0.1)

        _save_game_pitcher_cache(cache)

    return cache
