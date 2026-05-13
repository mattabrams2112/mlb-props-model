"""
Persistent pre-game ratings cache.
Saves ratings the first time they're calculated so they never change,
even across redeploys.
Key: date + player_id
"""
import os
import pandas as pd
from data_dir import data_path

CACHE_FILE   = data_path('ratings_cache.csv')
DATABASE_URL = os.environ.get('DATABASE_URL', '')


def _get_engine():
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine
        url = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        return create_engine(url)
    except Exception:
        return None


def _load() -> pd.DataFrame:
    engine = _get_engine()
    if engine:
        try:
            return pd.read_sql('SELECT * FROM ratings_cache', engine)
        except Exception:
            pass
    if os.path.exists(CACHE_FILE):
        try:
            df = pd.read_csv(CACHE_FILE, dtype={'player_id': str})
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=['date', 'player_id', 'rating', 'grade', 'projected'])


def _save(df: pd.DataFrame):
    engine = _get_engine()
    if engine:
        try:
            df.to_sql('ratings_cache', engine, if_exists='replace', index=False)
            return
        except Exception:
            pass
    df.to_csv(CACHE_FILE, index=False)


def get_cached_rating(game_date: str, player_id: int):
    """Returns cached (rating, grade, projected) or None if not saved yet."""
    df = _load()
    if df.empty:
        return None
    row = df[(df['date'] == game_date) & (df['player_id'] == str(player_id))]
    if row.empty:
        return None
    r = row.iloc[0]
    return int(r['rating']), str(r['grade']), float(r['projected'])


def save_rating(game_date: str, player_id: int, rating: int, grade: str, projected: float):
    """Save a rating — only if not already saved for this date+player."""
    df = _load()
    key = (df['date'] == game_date) & (df['player_id'] == str(player_id))
    if not df.empty and key.any():
        return  # already saved, don't overwrite
    new_row = pd.DataFrame([{
        'date':      game_date,
        'player_id': str(player_id),
        'rating':    rating,
        'grade':     grade,
        'projected': projected,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save(df)
