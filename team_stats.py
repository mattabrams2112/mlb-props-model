"""
Team recent scoring — avg runs per game over last N games.
High-scoring teams = more RBI/run opportunities.
"""
import os
import statsapi
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from data_dir import data_path

CACHE_FILE   = data_path('cache_team_scoring.csv')
DATABASE_URL = os.environ.get('DATABASE_URL', '')

TEAM_IDS = {
    'ARI':109,'ATL':144,'BAL':110,'BOS':111,'CHC':112,'CWS':145,'CIN':113,
    'CLE':114,'COL':115,'DET':116,'HOU':117,'KC':118,'LAA':108,'LAD':119,
    'MIA':146,'MIL':158,'MIN':142,'NYM':121,'NYY':147,'OAK':133,'PHI':143,
    'PIT':134,'SD':135,'SEA':136,'SF':137,'STL':138,'TB':139,'TEX':140,
    'TOR':141,'WSH':120,
}


LEAGUE_AVG_ERRORS = 100  # approx errors per team per season


def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        df = pd.read_csv(CACHE_FILE, dtype={'key': str})
        if df.empty or 'key' not in df.columns:
            return {}
        return df.set_index('key').to_dict('index')
    except Exception:
        return {}


def _save_cache(cache: dict):
    pd.DataFrame([{'key': k, **v} for k, v in cache.items()]).to_csv(CACHE_FILE, index=False)


@st.cache_data(show_spinner=False, ttl=3600)
def get_team_recent_scoring(team_abbr: str, n_games: int = 7) -> dict:
    """Returns team's avg runs/game and runs/game allowed over last N games."""
    defaults = {'team_runs_avg': 4.5, 'team_runs_allowed_avg': 4.5}
    team_id  = TEAM_IDS.get(team_abbr.upper())
    if not team_id:
        return defaults

    cache = _load_cache()
    today = datetime.now().strftime('%Y-%m-%d')
    key   = f'{team_abbr}_{today}'
    if key in cache:
        return cache[key]

    try:
        end   = datetime.now()
        start = end - timedelta(days=21)  # look back 3 weeks to find N games
        games = statsapi.schedule(
            team=team_id,
            start_date=start.strftime('%m/%d/%Y'),
            end_date=end.strftime('%m/%d/%Y'),
            sportId=1
        )
        completed = [g for g in games if g.get('status') in ('Final', 'Game Over')][-n_games:]
        if not completed:
            return defaults

        runs, runs_allowed = [], []
        for g in completed:
            if g.get('home_id') == team_id:
                runs.append(int(g.get('home_score', 0) or 0))
                runs_allowed.append(int(g.get('away_score', 0) or 0))
            else:
                runs.append(int(g.get('away_score', 0) or 0))
                runs_allowed.append(int(g.get('home_score', 0) or 0))

        result = {
            'team_runs_avg':         round(sum(runs) / len(runs), 2),
            'team_runs_allowed_avg': round(sum(runs_allowed) / len(runs_allowed), 2),
        }
        cache[key] = result
        _save_cache(cache)
        return result
    except Exception:
        return defaults


@st.cache_data(show_spinner=False, ttl=10800)   # 3h — season numbers move slowly
def get_team_season_strength(team_abbr: str) -> dict:
    """
    Season-long team strength from the full schedule:
      pyth_wpct      — Pythagorean expectation RS^1.83/(RS^1.83+RA^1.83),
                       a steadier "true talent" signal than recent form
      home_wpct / away_wpct — actual win% splits for venue-specific home field
    One schedule call per team, cached in-process and on disk per day.
    """
    defaults = {'pyth_wpct': 0.500, 'home_wpct': 0.540, 'away_wpct': 0.460,
                'rs_pg': 4.5, 'ra_pg': 4.5, 'games': 0}
    team_id = TEAM_IDS.get(team_abbr.upper())
    if not team_id:
        return defaults

    cache = _load_cache()
    from eastern_time import today_str_et
    today = today_str_et()
    key   = f'strength_{team_abbr}_{today}'
    if key in cache:
        return cache[key]

    try:
        season = int(today[:4])
        games  = statsapi.schedule(
            team=team_id,
            start_date=f'03/15/{season}',
            end_date=datetime.strptime(today, '%Y-%m-%d').strftime('%m/%d/%Y'),
            sportId=1
        )
        rs = ra = 0
        hw = hl = aw = al = 0
        n  = 0
        for g in games:
            if g.get('status') not in ('Final', 'Game Over', 'Completed Early'):
                continue
            if g.get('game_type') not in ('R', None):   # regular season only
                continue
            hs = int(g.get('home_score', 0) or 0)
            as_ = int(g.get('away_score', 0) or 0)
            n += 1
            if g.get('home_id') == team_id:
                rs += hs; ra += as_
                if hs > as_: hw += 1
                else:        hl += 1
            else:
                rs += as_; ra += hs
                if as_ > hs: aw += 1
                else:        al += 1
        if n < 15:   # too early in the season to trust
            return defaults
        exp  = 1.83
        pyth = (rs ** exp) / ((rs ** exp) + (ra ** exp)) if (rs + ra) > 0 else 0.5
        result = {
            'pyth_wpct': round(pyth, 4),
            'home_wpct': round(hw / (hw + hl), 4) if (hw + hl) >= 5 else 0.540,
            'away_wpct': round(aw / (aw + al), 4) if (aw + al) >= 5 else 0.460,
            'rs_pg':     round(rs / n, 2),
            'ra_pg':     round(ra / n, 2),
            'games':     n,
        }
        cache[key] = result
        _save_cache(cache)
        return result
    except Exception:
        return defaults


@st.cache_data(show_spinner=False, ttl=86400)
def get_team_defense_rating(team_abbr: str, season: int = None) -> dict:
    """
    Returns a defensive rating score.
    Lower errors + better fielding pct = higher score (better for opposing batters... wait)
    Actually: WORSE defense = MORE hits = good for batter.
    Score > 1.0 = defense is bad (more errors than avg = good for batter)
    Score < 1.0 = defense is good (fewer errors = bad for batter)
    """
    if season is None:
        from datetime import datetime
        season = datetime.now().year
    defaults = {'def_errors_rate': 1.0, 'def_rating': 0.0}
    team_id  = TEAM_IDS.get(team_abbr.upper())
    if not team_id:
        return defaults
    cache = _load_cache()
    key   = f'def_{team_abbr}_{season}'
    if key in cache:
        return cache[key]
    try:
        data = statsapi.get('stats', {
            'stats': 'season', 'group': 'fielding',
            'teamId': team_id, 'season': season, 'playerPool': 'all',
        })
        # Sum up team errors from all fielder splits
        total_errors = 0
        for split in (data.get('stats') or [{}])[0].get('splits', []):
            total_errors += int(split.get('stat', {}).get('errors', 0) or 0)
        # Rate relative to league average — >1.0 means more errors (bad defense)
        rate = round(total_errors / max(LEAGUE_AVG_ERRORS, 1), 3)
        # Rating for batter: bad defense (+) good for batter, good defense (-) bad
        def_bonus = round((rate - 1.0) * 5, 2)  # -5 to +5 pts range
        result = {'def_errors_rate': rate, 'def_rating': def_bonus}
    except Exception:
        result = defaults
    cache[key] = result
    _save_cache(cache)
    return result
