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


def no_vig_prob(over_odds, under_odds):
    """
    Market's true P(over) with the book's margin removed.

    Raw American odds carry vig, so the two sides sum to >100% (e.g. -125/+105
    implies 55.6% + 48.8% = 104.3%). Normalising by that total strips the margin
    and leaves what the market actually believes: here 53.3%.

    Both sides are required — one price alone carries no information about how
    much of it is margin. Returns None when the pair is missing or degenerate.
    """
    try:
        po = american_to_prob(int(over_odds))
        pu = american_to_prob(int(under_odds))
    except (TypeError, ValueError):
        return None
    total = po + pu
    if total <= 0 or not (0.90 < total < 1.35):   # implausible pair — bad data
        return None
    return round(po / total, 4)


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
                    # Pair Over/Under per player at the same line WITHIN a single
                    # book — vig can only be stripped from two sides of the same
                    # market. Mixing books would mix margins and prices.
                    book_sides: dict = {}
                    for outcome in mkt.get('outcomes', []):
                        side   = outcome.get('name')
                        player = outcome.get('description', '')
                        line   = outcome.get('point')
                        odds   = outcome.get('price')
                        if not player or line is None or side not in ('Over', 'Under'):
                            continue
                        key = (player, float(line))
                        book_sides.setdefault(key, {})[side] = odds
                        if side == 'Over':
                            if player not in raw:
                                raw[player] = {'lines': [], 'odds': [], 'novig': []}
                            raw[player]['lines'].append(float(line))
                            if odds is not None:
                                raw[player]['odds'].append(int(odds))
                    for (player, _ln), sides in book_sides.items():
                        nv = no_vig_prob(sides.get('Over'), sides.get('Under'))
                        if nv is not None and player in raw:
                            raw[player]['novig'].append(nv)
            lines = {}
            for player, data_p in raw.items():
                consensus_line = round(sum(data_p['lines']) / len(data_p['lines']) * 2) / 2
                avg_odds = int(sum(data_p['odds']) / len(data_p['odds'])) if data_p['odds'] else -110
                nv_list  = data_p.get('novig') or []
                lines[player] = {
                    'line':       consensus_line,
                    'over_odds':  avg_odds,
                    # Consensus no-vig P(over) across books. None when no book
                    # returned both sides — logged as a benchmark only, never fed
                    # into ratings, projections, or bet selection.
                    'novig_prob': round(sum(nv_list) / len(nv_list), 4) if nv_list else None,
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


# ── Moneylines (game winners) ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=900)   # cache 15 min
def get_moneylines() -> dict:
    """
    Fetch h2h (moneyline) odds for ALL of today's games in ONE request —
    unlike player props, the /odds endpoint covers every event at once, so
    this costs 1 quota request total.

    Returns {team_name_key: event} where each event is
      {'home_team', 'away_team', 'home_ml', 'away_ml', 'home_prob', 'away_prob'}
    keyed by both teams' full names and last words ("Yankees"). home_prob /
    away_prob are DEVIGGED implied probabilities (sum to 1.0).
    """
    if not ODDS_API_KEY:
        return {}
    try:
        resp = requests.get(
            f'{BASE_URL}/sports/{SPORT}/odds',
            params={
                'apiKey':     ODDS_API_KEY,
                'regions':    'us',
                'markets':    'h2h',
                'oddsFormat': 'american',
                'bookmakers': 'draftkings,fanduel,betmgm',
            },
            timeout=10
        )
        resp.raise_for_status()
        result = {}
        for event in resp.json():
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            h_odds, a_odds = [], []
            for book in event.get('bookmakers', []):
                for mkt in book.get('markets', []):
                    if mkt.get('key') != 'h2h':
                        continue
                    for out in mkt.get('outcomes', []):
                        price = out.get('price')
                        if price is None:
                            continue
                        if out.get('name') == home:
                            h_odds.append(int(price))
                        elif out.get('name') == away:
                            a_odds.append(int(price))
            if not h_odds or not a_odds:
                continue
            home_ml = int(sum(h_odds) / len(h_odds))
            away_ml = int(sum(a_odds) / len(a_odds))
            p_h = american_to_prob(home_ml)
            p_a = american_to_prob(away_ml)
            s   = p_h + p_a
            ev  = {
                'home_team': home, 'away_team': away,
                'home_ml': home_ml, 'away_ml': away_ml,
                'home_prob': round(p_h / s, 4) if s else 0.5,
                'away_prob': round(p_a / s, 4) if s else 0.5,
            }
            for team in (home, away):
                result[team] = ev
                words = team.split()
                if words:
                    result.setdefault(words[-1], ev)
        return result
    except Exception:
        return {}


# ── Game totals → implied team runs ───────────────────────────────────────────

def _margin_from_win_prob(p_home: float, sigma: float = 4.0) -> float:
    """
    Expected home run margin implied by a devigged home win probability.

    Treats the run margin as normal with sd = sigma, so margin = z(p) * sigma.

    sigma=4.0 is EMPIRICALLY VALIDATED, not guessed: real MLB finals sampled
    2026-06-18 -> 2026-08-04 give a run-margin sd of 4.056 (mean +0.130, i.e. the
    expected small home edge) and a mean game total of 8.796, which lines up with
    the totals books actually post. A 60% favourite comes out at +1.0 runs.

    Known limits: MLB margins aren't perfectly normal (discrete, never 0, mildly
    skewed), and a book posting team totals directly would price in lineup and
    park asymmetries this can't see. It only splits a total the market already
    set, so the total carries most of the signal and the ranking between the two
    sides — which is what matters here — is robust to small margin errors.
    """
    try:
        from statistics import NormalDist
        p = max(0.02, min(0.98, float(p_home)))
        return NormalDist().inv_cdf(p) * sigma
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False, ttl=900)   # cache 15 min
def get_game_totals() -> dict:
    """
    Every game's over/under total AND moneyline in ONE request, converted into
    each team's IMPLIED RUN TOTAL for tonight.

    Why this matters for H+R+RBI: two of its three components (runs, RBI) are
    run-scoring events, so a team's expected runs is the most direct driver of
    the stat. The model otherwise uses a SEASON average, which is the same number
    every day; the market's number is specific to this game and already prices in
    the starter, park, weather, bullpen, lineup and late scratches.

    implied_total = total/2 +/- margin/2, where margin comes from the devigged
    moneyline. Returns {team_key: {...}} keyed by full name and last word,
    matching get_moneylines()'s convention.
    """
    if not ODDS_API_KEY:
        return {}
    try:
        resp = requests.get(
            f'{BASE_URL}/sports/{SPORT}/odds',
            params={
                'apiKey':     ODDS_API_KEY,
                'regions':    'us',
                'markets':    'h2h,totals',
                'oddsFormat': 'american',
                'bookmakers': 'draftkings,fanduel,betmgm',
            },
            timeout=10
        )
        resp.raise_for_status()
        result = {}
        for event in resp.json():
            home = event.get('home_team', '')
            away = event.get('away_team', '')
            totals, h_odds, a_odds = [], [], []
            for book in event.get('bookmakers', []):
                for mkt in book.get('markets', []):
                    key = mkt.get('key')
                    for out in mkt.get('outcomes', []):
                        if key == 'totals':
                            pt = out.get('point')
                            if pt is not None and out.get('name') == 'Over':
                                totals.append(float(pt))
                        elif key == 'h2h':
                            price = out.get('price')
                            if price is None:
                                continue
                            if out.get('name') == home:
                                h_odds.append(int(price))
                            elif out.get('name') == away:
                                a_odds.append(int(price))
            if not totals:
                continue
            game_total = round(sum(totals) / len(totals), 2)

            # Split the total using the devigged moneyline; even 50/50 if absent.
            if h_odds and a_odds:
                p_h = american_to_prob(int(sum(h_odds) / len(h_odds)))
                p_a = american_to_prob(int(sum(a_odds) / len(a_odds)))
                s   = p_h + p_a
                p_home = (p_h / s) if s else 0.5
            else:
                p_home = 0.5
            margin     = _margin_from_win_prob(p_home)
            home_total = round(game_total / 2 + margin / 2, 2)
            away_total = round(game_total / 2 - margin / 2, 2)

            for team, own, opp in ((home, home_total, away_total),
                                   (away, away_total, home_total)):
                ev = {
                    'game_total':        game_total,
                    'implied_total':     own,
                    'opp_implied_total': opp,
                    'home_win_prob':     round(p_home, 4),
                }
                result[team] = ev
                words = team.split()
                if words:
                    result.setdefault(words[-1], ev)
        return result
    except Exception:
        return {}


def get_team_implied_total(team_name: str) -> float | None:
    """Implied run total for a team tonight, or None when unavailable."""
    if not team_name:
        return None
    tot = get_game_totals()
    if not tot:
        return None
    ev = tot.get(team_name) or tot.get(str(team_name).split()[-1])
    return ev.get('implied_total') if ev else None


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
