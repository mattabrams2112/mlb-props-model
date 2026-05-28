"""
Fetches player game logs, prioritizing current-season MLB data.

Priority order:
  1. Current-season MLB — use exclusively if >= MIN_GAMES
  2. Current-season AAA (scaled) — for callups still building MLB sample
  3. Prior-season MLB tail — only as many games as needed, not the full year
  4. Prior-season AAA tail — last resort for players with very limited history

Stats from minor leagues are scaled down to approximate MLB production:
  AAA: 0.85x  (talent gap ~15%)
  AA:  0.75x  (talent gap ~25%)
"""
import requests as _req
import pandas as pd
from datetime import datetime

MLB_API   = 'https://statsapi.mlb.com/api/v1'
MIN_GAMES = 25
AAA_SCALE = 0.85
AA_SCALE  = 0.75


def _fetch_mlb_rows(player_id: int, season: int) -> list:
    try:
        resp = _req.get(
            f'{MLB_API}/people/{player_id}/stats',
            params={'stats': 'gameLog', 'group': 'hitting', 'season': season},
            timeout=15,
        )
        resp.raise_for_status()
        splits = (resp.json().get('stats') or [{}])[0].get('splits', [])
        rows = []
        for s in splits:
            stat = s.get('stat', {}); gi = s.get('game', {})
            ih   = s.get('isHome', True)
            pt   = s.get('team', {}).get('abbreviation', '')
            op   = s.get('opponent', {}).get('abbreviation', '')
            rows.append({
                'player_id': player_id, 'season': season,
                'date':      gi.get('gameDate', s.get('date', '')),
                'game_pk':   str(gi.get('gamePk', '')),
                'opponent':  op, 'home_team': pt if ih else op,
                'is_home':   int(ih),
                'ab':  int(stat.get('atBats', 0)),
                'h':   int(stat.get('hits', 0)),
                'r':   int(stat.get('runs', 0)),
                'rbi': int(stat.get('rbi', 0)),
                'd':   int(stat.get('doubles', 0)),
                't':   int(stat.get('triples', 0)),
                'hr':  int(stat.get('homeRuns', 0)),
                'bb':  int(stat.get('baseOnBalls', 0)),
                'k':   int(stat.get('strikeOuts', 0)),
                'sb':  int(stat.get('stolenBases', 0)),
            })
        return rows
    except Exception:
        return []


def _fetch_milb_rows(player_id: int, season: int,
                     sport_id: int = 11, scale: float = AAA_SCALE) -> list:
    """Fetch minor league logs and scale counting stats to MLB equivalent."""
    try:
        resp = _req.get(
            f'{MLB_API}/people/{player_id}/stats',
            params={'stats': 'gameLog', 'group': 'hitting',
                    'season': season, 'sportId': sport_id},
            timeout=15,
        )
        resp.raise_for_status()
        splits = (resp.json().get('stats') or [{}])[0].get('splits', [])
        rows = []
        for s in splits:
            stat = s.get('stat', {}); gi = s.get('game', {})
            ih   = s.get('isHome', True)
            ab   = int(stat.get('atBats', 0))
            if not ab:
                continue
            rows.append({
                'player_id': player_id, 'season': season,
                'date':      gi.get('gameDate', s.get('date', '')),
                'game_pk':   str(gi.get('gamePk', '')),
                'opponent':  '', 'home_team': '',
                'is_home':   int(ih),
                'ab':  ab,
                'h':   round(int(stat.get('hits', 0))       * scale),
                'r':   round(int(stat.get('runs', 0))        * scale),
                'rbi': round(int(stat.get('rbi', 0))         * scale),
                'd':   round(int(stat.get('doubles', 0))     * scale),
                't':   round(int(stat.get('triples', 0))     * scale),
                'hr':  round(int(stat.get('homeRuns', 0))    * scale),
                'bb':  int(stat.get('baseOnBalls', 0)),
                'k':   int(stat.get('strikeOuts', 0)),
                'sb':  int(stat.get('stolenBases', 0)),
            })
        return rows
    except Exception:
        return []


def _to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df[df['ab'] > 0].reset_index(drop=True)


def fetch_player_logs(player_id: int) -> pd.DataFrame:
    """
    Returns game logs for a player, current season prioritized.
    Only pulls prior-season games as needed to reach MIN_GAMES.
    """
    current_year = datetime.now().year

    # 1. Current-season MLB — best case, use exclusively
    cur_rows = _fetch_mlb_rows(player_id, current_year)
    cur_df   = _to_df(cur_rows)
    if len(cur_df) >= MIN_GAMES:
        return cur_df

    # 2. Supplement with current-season MiLB for callups
    milb_cur = (
        _fetch_milb_rows(player_id, current_year, sport_id=11, scale=AAA_SCALE) +
        _fetch_milb_rows(player_id, current_year, sport_id=12, scale=AA_SCALE)
    )
    with_milb = _to_df(cur_rows + milb_cur)
    if len(with_milb) >= MIN_GAMES:
        return with_milb

    # 3. Pull only as many games as needed from prior MLB season
    needed    = MIN_GAMES - len(with_milb)
    prev_rows = _fetch_mlb_rows(player_id, current_year - 1)
    prev_df   = _to_df(prev_rows)
    prev_tail = prev_df.tail(needed) if not prev_df.empty else pd.DataFrame()

    combined = pd.concat([prev_tail, with_milb], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = (combined.dropna(subset=['date'])
                        .sort_values('date')
                        .reset_index(drop=True))
    combined = combined[combined['ab'] > 0].reset_index(drop=True)
    if len(combined) >= MIN_GAMES:
        return combined

    # 4. Last resort — prior season MiLB tail
    needed2        = MIN_GAMES - len(combined)
    milb_prev      = _to_df(
        _fetch_milb_rows(player_id, current_year - 1, sport_id=11, scale=AAA_SCALE)
    )
    milb_prev_tail = milb_prev.tail(needed2) if not milb_prev.empty else pd.DataFrame()

    final = pd.concat([milb_prev_tail, combined], ignore_index=True)
    final['date'] = pd.to_datetime(final['date'], errors='coerce')
    final = final.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return final[final['ab'] > 0].reset_index(drop=True)
