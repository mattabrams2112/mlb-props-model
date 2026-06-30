"""
Full play log — tracks EVERY HRR play regardless of rating.
Used for analytics to find profitable key numbers.
Only fetches actuals for PAST days (never today until tomorrow).
"""
import os
import statsapi
import requests
import pandas as pd
from datetime import datetime
from data_dir import data_path

LOG_FILE     = data_path('full_play_log.csv')
DATABASE_URL = os.environ.get('DATABASE_URL', '')
COLS = ['date', 'player', 'team', 'rating', 'grade', 'projected', 'base_proj',
        'line', 'over_odds', 'actual', 'result', 'vs_pitcher', 'is_home', 'pitcher_throws']


def _get_engine():
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine
        url = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        url = url.replace('postgres://', 'postgresql://', 1)
        if '?' not in url:
            url += '?sslmode=require'
        elif 'sslmode' not in url:
            url += '&sslmode=require'
        return create_engine(url, connect_args={'connect_timeout': 10})
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
             projected: float, base_proj: float = None,
             line: float = None, over_odds: int = None,
             vs_pitcher: str = '', is_home: bool = True,
             game_date: str = None, game_started: bool = False,
             pitcher_throws: str = ''):
    """Log a play. Updates rating/projection only if game hasn't started yet."""
    df    = load_all()
    today = game_date or datetime.now().strftime('%Y-%m-%d')
    existing = (not df.empty and
                ((df['date'].astype(str) == today) & (df['player'] == player)))
    if existing.any():
        idx = df[existing].index[0]
        # Only update if today, game hasn't started yet, and no actual recorded
        is_today = (today == datetime.now().strftime('%Y-%m-%d'))
        if is_today and not game_started and str(df.at[idx, 'actual']).strip() in ('', 'nan'):
            df.at[idx, 'rating']     = rating
            df.at[idx, 'grade']      = grade
            df.at[idx, 'projected']  = projected
            df.at[idx, 'base_proj']  = base_proj if base_proj is not None else ''
            df.at[idx, 'vs_pitcher'] = vs_pitcher
            save_all(df)
        return
    new_row = pd.DataFrame([{
        'date':      today,
        'player':    player,
        'team':      team,
        'rating':    rating,
        'grade':     grade,
        'projected': projected,
        'base_proj': base_proj if base_proj is not None else '',
        'line':      line or '',
        'over_odds': over_odds or '',
        'actual':    '',
        'result':    '',
        'vs_pitcher':     vs_pitcher,
        'is_home':        int(is_home),
        'pitcher_throws': pitcher_throws,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_all(df)


def _get_boxscore_stats_for_date(game_date: str) -> dict:
    """
    Fetch H+R+RBI for every player who played on a given date.
    Returns {player_name_lower: hrr} using boxscores — fast, one call per game.
    """
    result = {}
    try:
        date_fmt = datetime.strptime(game_date, '%Y-%m-%d').strftime('%m/%d/%Y')
        games = statsapi.schedule(date=date_fmt, sportId=1)
        for game in games:
            if game.get('status') not in ('Final', 'Game Over', 'Completed Early'):
                continue
            try:
                box = statsapi.boxscore_data(game['game_id'])
                for side in ('home', 'away'):
                    for _, pdata in box.get(side, {}).get('players', {}).items():
                        name = pdata.get('person', {}).get('fullName', '').lower().strip()
                        if not name:
                            continue
                        stat = pdata.get('stats', {}).get('batting', {})
                        h   = int(stat.get('hits', 0) or 0)
                        r   = int(stat.get('runs', 0) or 0)
                        rbi = int(stat.get('rbi', 0) or 0)
                        result[name] = h + r + rbi
            except Exception:
                pass
    except Exception:
        pass
    return result


def update_actuals() -> int:
    """
    Fetch actuals for all pending plays from past days ONLY (never today).
    Uses boxscores — one API call per game, much faster than per-player lookup.
    """
    df = load_all()
    if df.empty:
        return 0

    today   = datetime.now().strftime('%Y-%m-%d')
    updated = 0

    # Only process past days — never today (games may still be in progress)
    pending = df[
        (df['actual'].astype(str).str.strip().isin(['', 'nan'])) &
        (df['date'].astype(str).str[:10] < today)
    ]
    if pending.empty:
        return 0

    # Group by date and fetch boxscores once per date
    for game_date in pending['date'].astype(str).str[:10].unique():
        player_stats = _get_boxscore_stats_for_date(game_date)
        if not player_stats:
            continue

        date_rows = df[df['date'].astype(str).str[:10] == game_date]
        for i in date_rows.index:
            row = df.loc[i]
            if str(row.get('actual', '')).strip() not in ('', 'nan'):
                continue
            player_lower = str(row.get('player', '')).lower().strip()
            hrr = player_stats.get(player_lower)
            # Partial last-name match fallback
            if hrr is None:
                parts = player_lower.split()
                last  = parts[-1] if parts else ''
                for k, v in player_stats.items():
                    if last and last in k:
                        hrr = v
                        break
            if hrr is not None:
                df.at[i, 'actual'] = str(hrr)
                line_val = str(row.get('line', '')).strip()
                # Only set W/L if a real line was recorded — no line means stay pending
                if line_val and line_val not in ('nan', ''):
                    line = float(line_val)
                    if game_date < today:
                        df.at[i, 'result'] = 'W' if hrr > line else 'L'
                updated += 1

    if updated:
        save_all(df)
    return updated
