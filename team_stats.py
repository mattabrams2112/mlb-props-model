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
