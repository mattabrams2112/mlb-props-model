"""
Full play log — tracks EVERY HRR play regardless of rating.
Used for analytics to find profitable key numbers.
"""
import os
import pandas as pd
from datetime import datetime
from data_dir import data_path

LOG_FILE     = data_path('full_play_log.csv')
DATABASE_URL = os.environ.get('DATABASE_URL', '')
COLS = ['date', 'player', 'team', 'rating', 'grade', 'projected',
        'line', 'over_odds', 'actual', 'result', 'vs_pitcher', 'is_home']


def _get_engine():
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine
        url = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        return create_engine(url)
    except Exception:
        return None


def load_all() -> pd.DataFrame:
    engine = _get_engine()
    if engine:
        try:
            df = pd.read_sql('SELECT * FROM full_play_log ORDER BY date DESC', engine)
            for c in COLS:
                if c not in df.columns:
                    df[c] = ''
            return df
        except Exception:
            pass
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save_all(df: pd.DataFrame):
    engine = _get_engine()
    if engine:
        try:
            df.to_sql('full_play_log', engine, if_exists='replace', index=False)
            return
        except Exception:
            pass
    df.to_csv(LOG_FILE, index=False)


def log_play(player: str, team: str, rating: int, grade: str,
             projected: float, line: float = None, over_odds: int = None,
             vs_pitcher: str = '', is_home: bool = True,
             game_date: str = None):
    """Log a play. Skips if already logged for this date+player."""
    df    = load_all()
    today = game_date or datetime.now().strftime('%Y-%m-%d')
    if not df.empty and ((df['date'] == today) & (df['player'] == player)).any():
        return
    new_row = pd.DataFrame([{
        'date':      today,
        'player':    player,
        'team':      team,
        'rating':    rating,
        'grade':     grade,
        'projected': projected,
        'line':      line or '',
        'over_odds': over_odds or '',
        'actual':    '',
        'result':    '',
        'vs_pitcher': vs_pitcher,
        'is_home':   int(is_home),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_all(df)


def update_actuals():
    """Fetch actual H+R+RBI for all pending plays and compute W/L."""
    import requests
    import statsapi

    df    = load_all()
    if df.empty:
        return 0

    today = datetime.now().strftime('%Y-%m-%d')
    updated = 0

    for i, row in df.iterrows():
        if str(row.get('actual', '')).strip() not in ('', 'nan'):
            continue
        game_date = str(row.get('date', ''))[:10]
        # Only fetch actuals for completed past days — never today
        if game_date >= today:
            continue
        try:
            # Use direct API for player lookup to avoid statsapi hanging
            lookup_resp = requests.get(
                'https://statsapi.mlb.com/api/v1/people/search',
                params={'names': row['player'], 'sportId': 1},
                timeout=8
            )
            people = lookup_resp.json().get('people', []) if lookup_resp.ok else []
            if not people:
                # Fallback to statsapi
                players = statsapi.lookup_player(row['player'])
                people = [{'id': players[0]['id']}] if players else []
            if not people:
                continue
            pid  = people[0]['id']
            year = game_date[:4]
            resp = requests.get(
                f'https://statsapi.mlb.com/api/v1/people/{pid}/stats',
                params={'stats': 'gameLog', 'group': 'hitting', 'season': year},
                timeout=15
            )
            resp.raise_for_status()
            splits = (resp.json().get('stats') or [{}])[0].get('splits', [])
            for split in splits:
                gi = split.get('game', {})
                if gi.get('gameDate', split.get('date', ''))[:10] == game_date:
                    s   = split.get('stat', {})
                    hrr = int(s.get('hits',0)) + int(s.get('runs',0)) + int(s.get('rbi',0))
                    df.at[i, 'actual'] = hrr
                    try:
                        line_val = str(row.get('line', '')).strip()
                        line = float(line_val) if line_val and line_val != 'nan' else 1.5
                        df.at[i, 'result'] = 'W' if hrr > line else 'L'
                    except Exception:
                        pass
                    updated += 1
                    break
        except Exception:
            pass

    if updated:
        save_all(df)
    return updated
