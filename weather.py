import os
import re
import time
import statsapi
import pandas as pd

from data_dir import data_path
CACHE_FILE = data_path('weather_cache.csv')

# Park factors by team (runs, normalized: 1.00 = neutral)
PARK_FACTORS = {
    'COL': 1.15, 'CIN': 1.06, 'TEX': 1.05, 'BOS': 1.04,
    'NYY': 1.03, 'PHI': 1.03, 'MIL': 1.02, 'CWS': 1.01,
    'LAA': 1.01, 'ATL': 1.00, 'LAD': 1.00, 'NYM': 1.00,
    'TOR': 1.00, 'BAL': 1.00, 'WSH': 0.99, 'KC':  0.99,
    'CHC': 0.99, 'STL': 0.99, 'MIN': 0.99, 'CLE': 0.98,
    'SF':  0.98, 'ARI': 0.98, 'DET': 0.97, 'TB':  0.97,
    'HOU': 0.97, 'PIT': 0.96, 'MIA': 0.96, 'SD':  0.95,
    'SEA': 0.95, 'OAK': 0.96,
}
DEFAULT_PARK_FACTOR = 1.00


def get_park_factor(team_abbr: str) -> float:
    return PARK_FACTORS.get((team_abbr or '').upper(), DEFAULT_PARK_FACTOR)


def parse_wind(wind_str: str) -> tuple:
    """Return (speed_mph, direction_code).
    direction_code: +1 = blowing out (hitter-friendly), -1 = blowing in, 0 = neutral/crosswind
    """
    if not wind_str or 'calm' in wind_str.lower():
        return 0.0, 0

    m = re.search(r'(\d+)\s*mph', wind_str, re.IGNORECASE)
    speed = float(m.group(1)) if m else 0.0

    lower = wind_str.lower()
    if 'out' in lower:
        direction = 1
    elif 'in' in lower:
        direction = -1
    else:
        direction = 0

    return speed, direction


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        df = pd.read_csv(CACHE_FILE, dtype={'game_pk': str})
        if df.empty or 'game_pk' not in df.columns:
            return {}
        return df.set_index('game_pk').to_dict('index')
    except Exception:
        return {}


def _save_cache(cache: dict):
    if not cache:
        return
    rows = [{'game_pk': pk, **vals} for pk, vals in cache.items()]
    pd.DataFrame(rows).to_csv(CACHE_FILE, index=False)


def fetch_weather_for_games(game_pks: list, verbose: bool = True) -> pd.DataFrame:
    cache = _load_cache()
    str_pks = [str(pk) for pk in game_pks]
    missing = [pk for pk in str_pks if pk not in cache]

    if missing:
        if verbose:
            print(f"  Fetching weather for {len(missing)} games "
                  f"({len(str_pks) - len(missing)} already cached)...")
        for i, pk in enumerate(missing):
            try:
                data = statsapi.get('game', {'gamePk': pk, 'fields': 'gameData,weather'})
                w = data.get('gameData', {}).get('weather', {})
                wind_speed, wind_dir = parse_wind(w.get('wind', ''))
                temp_raw = w.get('temp', '')
                temp_m = re.search(r'(\d+)', temp_raw)
                cache[pk] = {
                    'temp_f': float(temp_m.group(1)) if temp_m else None,
                    'wind_speed': wind_speed,
                    'wind_dir': wind_dir,
                    'condition': w.get('condition', ''),
                }
            except Exception:
                cache[pk] = {'temp_f': None, 'wind_speed': 0.0, 'wind_dir': 0, 'condition': ''}

            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(missing)} fetched...")
            time.sleep(0.1)

        _save_cache(cache)

    rows = []
    for pk in str_pks:
        entry = dict(cache.get(pk, {'temp_f': None, 'wind_speed': 0.0, 'wind_dir': 0, 'condition': ''}))
        entry['game_pk'] = pk
        rows.append(entry)

    return pd.DataFrame(rows)
