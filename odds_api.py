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
    """Returns {home_team_abbr: event_id} for today's MLB games."""
    if not ODDS_API_KEY:
        return {}
    try:
        resp = requests.get(
            f'{BASE_URL}/sports/{SPORT}/events',
            params={'apiKey': ODDS_API_KEY, 'dateFormat': 'iso'},
            timeout=10
        )
        resp.raise_for_status()
        today = datetime.now().strftime('%Y-%m-%d')
        result = {}
        for event in resp.json():
            if event.get('commence_time', '')[:10] == today:
                home_full = event.get('home_team', '')
                eid       = event.get('id', '')
                # Store by full name AND abbreviation for flexible matching
                result[home_full] = eid
                # Also store by last word (e.g. "Baltimore Orioles" -> "Orioles")
                words = home_full.split()
                if words:
                    result[words[-1]] = eid
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
            lines = {}
            for book in data.get('bookmakers', []):
                for mkt in book.get('markets', []):
                    for outcome in mkt.get('outcomes', []):
                        if outcome.get('name') == 'Over':
                            player = outcome.get('description', '')
                            line   = outcome.get('point')
                            odds   = outcome.get('price')
                            if player and line is not None and player not in lines:
                                lines[player] = {
                                    'line':      float(line),
                                    'over_odds': int(odds) if odds else -110,
                                }
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
