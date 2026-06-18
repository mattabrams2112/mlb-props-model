"""
Team bullpen stats — ERA, WHIP, K% for relief pitchers.
Used as a rating/projection modifier: a bad bullpen = more HRR opportunity in later innings.
Cached by team abbreviation + season.
"""
import os
import statsapi
import pandas as pd
from datetime import datetime

CURRENT_YEAR = datetime.now().year
CACHE_FILE   = 'cache_bullpen.csv'

# MLB team name → team ID mapping
TEAM_IDS = {
    'ARI': 109, 'ATL': 144, 'BAL': 110, 'BOS': 111, 'CHC': 112,
    'CWS': 145, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KC':  118, 'LAA': 108, 'LAD': 119, 'MIA': 146,
    'MIL': 158, 'MIN': 142, 'NYM': 121, 'NYY': 147, 'OAK': 133,
    'PHI': 143, 'PIT': 134, 'SD':  135, 'SEA': 136, 'SF':  137,
    'STL': 138, 'TB':  139, 'TEX': 140, 'TOR': 141, 'WSH': 120,
}

LEAGUE_BULLPEN = {'bp_era': 4.20, 'bp_whip': 1.30, 'bp_k_pct': 0.235}
_MEM_CACHE: dict = {}


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


def get_bullpen_stats(team_abbr: str, season: int = None) -> dict:
    """Returns team bullpen ERA, WHIP, K% for the given season."""
    if season is None:
        season = CURRENT_YEAR

    cache = _load_cache()
    key   = f'{team_abbr}_{season}'
    if key in cache:
        return cache[key]

    team_id = TEAM_IDS.get(team_abbr.upper())
    result  = LEAGUE_BULLPEN.copy()

    if team_id:
        try:
            data = statsapi.get('stats', {
                'stats':  'season',
                'group':  'pitching',
                'teamId': team_id,
                'season': season,
                'playerPool': 'all',
            })
            splits = (data.get('stats') or [{}])[0].get('splits', [])

            # Filter to relief pitchers: games started = 0 or GS < G/2
            era_list, whip_list, k_list, bf_list = [], [], [], []
            for s in splits:
                st = s.get('stat', {})
                gs = int(st.get('gamesStarted', 0) or 0)
                g  = int(st.get('gamesPlayed',  0) or 0)
                ip = float(st.get('inningsPitched', 0) or 0)
                if gs == 0 and ip >= 3:   # reliever with meaningful innings
                    try:
                        era_list.append(float(st.get('era',  4.20) or 4.20))
                        whip_list.append(float(st.get('whip', 1.30) or 1.30))
                        k   = float(st.get('strikeOuts',   0) or 0)
                        bf  = float(st.get('battersFaced', 1) or 1)
                        k_list.append(k / bf if bf > 0 else 0.235)
                        bf_list.append(bf)
                    except (ValueError, TypeError):
                        pass

            if era_list:
                result = {
                    'bp_era':   round(sum(era_list)  / len(era_list),  2),
                    'bp_whip':  round(sum(whip_list) / len(whip_list), 2),
                    'bp_k_pct': round(sum(k * b for k, b in zip(k_list, bf_list))
                                      / max(sum(bf_list), 1), 3),
                }
        except Exception:
            pass

    cache[key] = result
    _save_cache(cache)
    return result
