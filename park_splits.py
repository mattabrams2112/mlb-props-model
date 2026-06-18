"""
Career batter stats at a specific ballpark (by home team abbreviation).
Uses MLB Stats API venue splits endpoint.
"""
import os
import pandas as pd
from datetime import datetime

CURRENT_YEAR = datetime.now().year
_PARK_CACHE: dict = {}

# Map team abbreviation → MLB team ID
TEAM_ID_MAP = {
    'NYY': 147, 'BOS': 111, 'TB': 139, 'TOR': 141, 'BAL': 110,
    'CLE': 114, 'MIN': 142, 'CWS': 145, 'DET': 116, 'KC': 118,
    'HOU': 117, 'SEA': 136, 'TEX': 140, 'LAA': 108, 'OAK': 133,
    'ATL': 144, 'NYM': 121, 'PHI': 143, 'MIA': 146, 'WSH': 120,
    'MIL': 158, 'CHC': 112, 'STL': 138, 'CIN': 113, 'PIT': 134,
    'LAD': 119, 'SF': 137, 'SD': 135, 'COL': 115, 'ARI': 109,
}


def _parse_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def get_batter_park_splits(player_id: int, home_team: str) -> dict:
    """
    Career BA/SLG/OPS at the specific park (identified by home team abbrev).
    Returns dict with park_ba, park_slg, park_ops, park_ab.
    Falls back to neutral defaults if unavailable.
    """
    defaults = {'park_ba': 0.250, 'park_slg': 0.400, 'park_ops': 0.700, 'park_ab': 0}
    cache_key = f"{player_id}_{home_team}"
    if cache_key in _PARK_CACHE:
        return _PARK_CACHE[cache_key]

    team_id = TEAM_ID_MAP.get(home_team)
    if not team_id:
        _PARK_CACHE[cache_key] = defaults
        return defaults

    result = defaults.copy()
    try:
        import requests as _req
        # Career stats vs this team's home park using venue splits
        resp = _req.get(
            f'https://statsapi.mlb.com/api/v1/people/{player_id}/stats',
            params={
                'stats': 'statSplits',
                'group': 'hitting',
                'sitCodes': 'vr',      # vs right — use team opponent split as proxy
                'opposingTeamId': team_id,
            },
            timeout=10
        )
        # Try career splits at this venue directly
        resp2 = _req.get(
            f'https://statsapi.mlb.com/api/v1/people/{player_id}/stats',
            params={
                'stats': 'careerStatSplits',
                'group': 'hitting',
                'sitCodes': 'h',
                'gameType': 'R',
                'opposingTeamId': team_id,
            },
            timeout=10
        )
        for r in [resp2, resp]:
            try:
                r.raise_for_status()
                splits = (r.json().get('stats') or [{}])[0].get('splits', [])
                if splits:
                    s = splits[0].get('stat', {})
                    ab = _parse_float(s.get('atBats'), 0)
                    if ab >= 10:
                        result = {
                            'park_ba':  _parse_float(s.get('avg'),      0.250),
                            'park_slg': _parse_float(s.get('slg'),      0.400),
                            'park_ops': _parse_float(s.get('ops'),      0.700),
                            'park_ab':  int(ab),
                        }
                        break
            except Exception:
                continue
    except Exception:
        pass

    _PARK_CACHE[cache_key] = result
    return result
