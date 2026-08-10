"""
Career batter stats at a specific ballpark (by home team abbreviation).

Built from hitting GAME LOGS aggregated by ballpark, not from the splits
endpoint.

The previous implementation asked `careerStatSplits` with `sitCodes` plus
`opposingTeamId` and called the answer park history. That endpoint has no
venue parameter at all, and MLB ignores parameters it doesn't recognise rather
than erroring — so it returned the player's plain career split, byte-identical
for all 30 parks (verified 2026-08-10: one batter came back .272/.434 on 1127
AB whether the park was Toronto, Coors or Miami). The source even conceded it:
"vs right — use team opponent split as proxy". Dead since it shipped 2026-06-02.

Game logs don't carry a `venue` field either, but they carry `isHome` and
`opponent`, and the park follows from those: the batter's own team's park when
home, the opponent's when away. Aggregating AB/H/TB that way gives real
per-park numbers — the same batter now ranges .050 to .391 across 24 parks.
"""
import os
import pandas as pd
import requests
from collections import defaultdict
from datetime import datetime

CURRENT_YEAR = datetime.now().year

# player_id -> {park_team_id: {'ab','h','tb','g'}}. One fetch covers every park
# for that batter, so a full lineup costs one pass per player, not per park.
_PLAYER_PARK_CACHE: dict = {}

# How many seasons of history to aggregate. Enough to clear the sample gate at
# road parks (~50-90 AB over four years) without pulling a whole career.
SEASONS_BACK = 4

# Minimum AB at a park before we report it as signal.
#
# rating.py gates on `park_ab >= 20`, which is far too loose for a component
# worth -1.5 to +1.5 points: at 20 AB the standard error on batting average is
# ~.097, so a 2-for-20 stretch scores the maximum penalty on pure noise. Real
# examples from one batter's four-year logs: .100 on 20 AB at Toronto, .050 on
# 20 AB in Seattle. Both would have taken the full -1.5.
#
# Reporting park_ab=0 below this keeps rating.py's guard shut without touching
# the rating engine. 50 AB (SE ~.061) is still thin — a home park clears it
# easily and frequent division road parks usually do, distant parks rarely will.
# Treat this as a floor for turning the component on at all, not as evidence
# the signal is real; it has never been validated against outcomes.
MIN_PARK_AB = 50

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


def _load_player_parks(player_id: int) -> dict:
    """Aggregate a batter's regular-season game logs by ballpark."""
    if player_id in _PLAYER_PARK_CACHE:
        return _PLAYER_PARK_CACHE[player_id]

    totals = defaultdict(lambda: {'ab': 0, 'h': 0, 'tb': 0, 'g': 0})
    for season in range(CURRENT_YEAR - SEASONS_BACK + 1, CURRENT_YEAR + 1):
        try:
            resp = requests.get(
                f'https://statsapi.mlb.com/api/v1/people/{player_id}/stats',
                params={'stats': 'gameLog', 'group': 'hitting', 'season': season},
                timeout=15)
            resp.raise_for_status()
            splits = (resp.json().get('stats') or [{}])[0].get('splits', [])
        except Exception:
            continue

        for s in splits:
            if s.get('gameType') != 'R':      # regular season only
                continue
            # No venue field on these rows — derive the park from which side
            # the batter was on.
            park_id = ((s.get('team') or {}).get('id') if s.get('isHome')
                       else (s.get('opponent') or {}).get('id'))
            if not park_id:
                continue
            st = s.get('stat', {})
            t  = totals[int(park_id)]
            t['ab'] += int(_parse_float(st.get('atBats'), 0))
            t['h']  += int(_parse_float(st.get('hits'), 0))
            t['tb'] += int(_parse_float(st.get('totalBases'), 0))
            t['g']  += 1

    result = dict(totals)
    _PLAYER_PARK_CACHE[player_id] = result
    return result


def get_batter_park_splits(player_id: int, home_team: str) -> dict:
    """
    BA/SLG/OPS at the specific park (identified by home team abbrev), over the
    last SEASONS_BACK regular seasons.

    Returns park_ba, park_slg, park_ops, park_ab. park_ab stays 0 when the park
    is unknown or the sample is thin, which is what keeps rating.py's
    `if park_ab >= 20` guard shut rather than scoring noise.
    """
    defaults = {'park_ba': 0.250, 'park_slg': 0.400, 'park_ops': 0.700, 'park_ab': 0}

    team_id = TEAM_ID_MAP.get(home_team)
    if not team_id:
        return defaults

    try:
        parks = _load_player_parks(player_id)
    except Exception:
        return defaults

    t = parks.get(team_id)
    if not t or t['ab'] < MIN_PARK_AB:
        return defaults

    ba  = t['h']  / t['ab']
    slg = t['tb'] / t['ab']
    return {
        'park_ba':  round(ba, 3),
        'park_slg': round(slg, 3),
        # True OPS needs OBP, which the game-log aggregate doesn't carry.
        # rating.py only reads park_ba and park_slg, so this stays an estimate
        # and is not used for scoring.
        'park_ops': round(ba + slg, 3),
        'park_ab':  int(t['ab']),
    }
