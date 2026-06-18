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
        if '?' not in url:
            url += '?sslmode=require'
        elif 'sslmode' not in url:
            url += '&sslmode=require'
        return create_engine(url, connect_args={'connect_timeout': 10})
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


def clear_ratings_for_players(game_date: str, player_ids: list):
    """Delete cached ratings for a list of players on a given date."""
    df = _load()
    if df.empty:
        return
    ids = [str(p) for p in player_ids]
    mask = (df['date'] == game_date) & (df['player_id'].isin(ids))
    df = df[~mask].reset_index(drop=True)
    _save(df)


def get_cached_rating(game_date: str, player_id: int, opp_pitcher: str = None):
    """Returns cached (rating, grade, projected) or None if not saved yet.
    If opp_pitcher is given, only returns a match when vs_pitcher matches —
    ensures doubleheader Game 2 isn't served Game 1's cached rating."""
    df = _load()
    if df.empty:
        return None
    rows = df[(df['date'] == game_date) & (df['player_id'] == str(player_id))]
    if rows.empty:
        return None
    if opp_pitcher and 'vs_pitcher' in rows.columns:
        matched = rows[rows['vs_pitcher'].astype(str).str.strip() == opp_pitcher.strip()]
        if matched.empty:
            return None
        rows = matched
    r = rows.iloc[0]
    return int(r['rating']), str(r['grade']), float(r['projected'])


def save_rating(game_date: str, player_id: int, rating: int, grade: str,
                projected: float, player_name: str = '', team: str = '',
                vs_pitcher: str = ''):
    """Save a rating — only if not already saved for this date+player.
    Also auto-adds 75+ rated players to the tracker."""
    df = _load()
    key = (df['date'] == game_date) & (df['player_id'] == str(player_id))
    if not df.empty and key.any():
        return  # already saved, don't overwrite

    new_row = pd.DataFrame([{
        'date':        game_date,
        'player_id':   str(player_id),
        'rating':      rating,
        'grade':       grade,
        'projected':   projected,
        'player_name': player_name,
        'team':        team,
        'vs_pitcher':  vs_pitcher,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save(df)

    # Auto-add to tracker using current criteria
    _qualifies = rating >= 75
    if _qualifies and player_name:
        try:
            from tracker import add_predictions
            add_predictions([{
                'player':     player_name,
                'team':       team,
                'rating':     rating,
                'grade':      grade,
                'projected':  projected,
                'vs_pitcher': vs_pitcher,
            }], game_date=game_date)
        except Exception:
            pass
        # Mirror to full_play_log so Daily Results stays in sync with Tracker
        try:
            from full_tracker import log_play as _fpl_log
            _fpl_log(player=player_name, team=team, rating=rating, grade=grade,
                     projected=projected, vs_pitcher=vs_pitcher, game_date=game_date)
        except Exception:
            pass

    # Always update any existing tracker entry so recalculated ratings are reflected.
    # If the new rating no longer qualifies, the tracker display filter will hide the row.
    if player_name:
        try:
            from tracker import update_rating_if_exists
            update_rating_if_exists(player_name, game_date, rating, grade,
                                    projected, vs_pitcher)
        except Exception:
            pass
