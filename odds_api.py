"""
The Odds API integration — fetches MLB H+R+RBI over lines and odds.
Requires ODDS_API_KEY environment variable.
"""
import os
import math
import unicodedata
import requests
import streamlit as st
from datetime import datetime

ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')
BASE_URL     = 'https://api.the-odds-api.com/v4'
SPORT        = 'baseball_mlb'
# Try these markets in order — different books use different names
HRR_MARKETS  = ['batter_hits_runs_rbis', 'player_hits_runs_rbis', 'batter_total_hits_runs_rbis']


# ── Math helpers ──────────────────────────────────────────────────────────────

def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for Poisson(lam)."""
    if lam <= 0:
        return 1.0
    total = 0.0
    term  = math.exp(-lam)
    for i in range(int(k) + 1):
        total += term
        term  *= lam / (i + 1)
    return min(1.0, total)


def fair_probability(projection: float, line: float) -> float:
    """P(H+R+RBI > line) using Poisson(projection)."""
    k = int(line)   # e.g. line=1.5 → k=1 → P(X>1.5)=P(X>=2)=1-P(X<=1)
    return max(0.0, min(1.0, 1.0 - _poisson_cdf(k, max(projection, 0.01))))


def american_to_prob(odds: int) -> float:
    """American odds → implied probability (includes vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def prob_to_american(p: float) -> int:
    """Fair probability → American odds (no vig)."""
    p = max(0.01, min(0.99, p))
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)


def edge_rating_bonus(edge: float) -> float:
    """Rating points to add/subtract based on edge (model prob - book implied prob)."""
    if edge >= 0.15:   return 12.0
    if edge >= 0.10:   return 8.0
    if edge >= 0.05:   return 5.0
    if edge >= 0.02:   return 2.0
    if edge >= -0.05:  return 0.0
    if edge >= -0.10:  return -5.0
    return -10.0


# ── Name matching ─────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    return name.lower().replace('.', '').replace("'", '').replace('-', ' ').strip()


def match_player(target: str, candidates: list) -> str | None:
    """Return the best matching candidate name for target, or None."""
    norm_target = _normalize(target)
    for c in candidates:
        if _normalize(c) == norm_target:
            return c
    # Partial match — last name
    last = norm_target.split()[-1] if norm_target else ''
    for c in candidates:
        if last and last in _normalize(c):
            return c
    return None


# ── API calls ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=900)   # cache 15 min
def get_todays_event_ids() -> dict:
    """
    Returns {team_name: event_id} for today's (Eastern) MLB games, keyed by
    BOTH the home and away team — full name and last word — so a player on
    either side can find their event.

    The Odds API returns commence_time in UTC. MLB night games start 7-8pm ET
    which is 23:00-01:00 UTC, so comparing the raw UTC date against "today"
    drops evening games (and drops ALL games when the page is loaded after the
    UTC rollover). We convert each commence_time to Eastern before comparing.
    """
    if not ODDS_API_KEY:
        return {}
    try:
        resp = requests.get(
            f'{BASE_URL}/sports/{SPORT}/events',
            params={'apiKey': ODDS_API_KEY, 'dateFormat': 'iso'},
            timeout=10
        )
        resp.raise_for_status()

        from eastern_time import today_str_et
        try:
            from zoneinfo import ZoneInfo
            _ET = ZoneInfo('America/New_York')
        except ImportError:
            from datetime import timezone, timedelta
            _ET = timezone(timedelta(hours=-4))   # EDT — MLB season

        today = today_str_et()
        result = {}
        for event in resp.json():
            ct = event.get('commence_time', '')
            if not ct:
                continue
            try:
                dt      = datetime.fromisoformat(ct.replace('Z', '+00:00'))
                et_date = dt.astimezone(_ET).strftime('%Y-%m-%d')
            except Exception:
                et_date = ct[:10]
            if et_date != today:
                continue
            eid = event.get('id', '')
            for team in (event.get('home_team', ''), event.get('away_team', '')):
                if not team:
                    continue
                result[team] = eid
                words = team.split()
                if words:
                    result.setdefault(words[-1], eid)   # don't clobber a full-name match
        return result
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=900)
def get_hrr_lines(event_id: str) -> dict:
    """
    Returns {player_name: {'line': float, 'over_odds': int}} for one event.
    Tries multiple market names until one works.
    """
    if not ODDS_API_KEY or not event_id:
        return {}
    for market in HRR_MARKETS:
        try:
            resp = requests.get(
                f'{BASE_URL}/sports/{SPORT}/events/{event_id}/odds',
                params={
                    'apiKey':      ODDS_API_KEY,
                    'regions':     'us',
                    'markets':     market,
                    'oddsFormat':  'american',
                    'bookmakers':  'draftkings,fanduel,betmgm',
                },
                timeout=10
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            # Collect all lines per player across all bookmakers, then average
            raw: dict = {}
            for book in data.get('bookmakers', []):
                for mkt in book.get('markets', []):
                    for outcome in mkt.get('outcomes', []):
                        if outcome.get('name') == 'Over':
                            player = outcome.get('description', '')
                            line   = outcome.get('point')
                            odds   = outcome.get('price')
                            if player and line is not None:
                                if player not in raw:
                                    raw[player] = {'lines': [], 'odds': []}
                                raw[player]['lines'].append(float(line))
                                if odds is not None:
                                    raw[player]['odds'].append(int(odds))
            lines = {}
            for player, data_p in raw.items():
                consensus_line = round(sum(data_p['lines']) / len(data_p['lines']) * 2) / 2
                avg_odds = int(sum(data_p['odds']) / len(data_p['odds'])) if data_p['odds'] else -110
                lines[player] = {'line': consensus_line, 'over_odds': avg_odds}
            if lines:
                return lines
        except Exception:
            continue
    return {}


def get_player_line(player_name: str, event_id: str) -> dict | None:
    """Fetch line + odds for a specific player. Returns None if not found."""
    lines = get_hrr_lines(event_id)
    if not lines:
        return None
    matched = match_player(player_name, list(lines.keys()))
    if matched:
        entry   = lines[matched]
        return {
            'line':       entry['line'],
            'over_odds':  entry['over_odds'],
            'implied_prob': american_to_prob(entry['over_odds']),
        }
    return None


# ── API status / quota ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)   # cache 5 min
def get_api_status() -> dict:
    """
    Live Odds API health check. The /events endpoint is FREE (does not count
    against the monthly quota) but still returns the quota headers, so this
    tells us the key is valid and how many requests remain — without spending.
    """
    if not ODDS_API_KEY:
        return {'key_set': False, 'remaining': None, 'used': None,
                'error': 'No ODDS_API_KEY configured'}
    try:
        resp = requests.get(
            f'{BASE_URL}/sports/{SPORT}/events',
            params={'apiKey': ODDS_API_KEY, 'dateFormat': 'iso'},
            timeout=10
        )
        remaining = resp.headers.get('x-requests-remaining')
        used      = resp.headers.get('x-requests-used')
        if resp.status_code == 401:
            return {'key_set': True, 'remaining': remaining, 'used': used,
                    'error': 'Invalid API key (401)'}
        if resp.status_code == 429:
            return {'key_set': True, 'remaining': remaining, 'used': used,
                    'error': 'Quota exhausted (429)'}
        resp.raise_for_status()
        return {'key_set': True, 'remaining': remaining, 'used': used, 'error': None}
    except Exception as e:
        return {'key_set': True, 'remaining': None, 'used': None,
                'error': str(e)[:120]}


def render_api_status():
    """Render a one-line Odds API status caption (key / quota / errors)."""
    s = get_api_status()
    if not s['key_set']:
        st.caption('🔴 **Odds API:** no key configured — enter lines manually.')
        return
    if s['error']:
        st.caption(f'🔴 **Odds API:** {s["error"]} — enter lines manually.')
        return
    rem, used = s['remaining'], s['used']
    used_str  = f' · {used} used' if used else ''
    if rem is not None:
        try:
            rem_i = int(float(rem))
            dot = '🟢' if rem_i > 50 else '🟡' if rem_i > 0 else '🔴'
            st.caption(f'{dot} **Odds API:** {rem_i} requests remaining this month{used_str}')
            return
        except ValueError:
            pass
    st.caption('🟢 **Odds API:** connected')
