"""
Core tracker logic.
- If DATABASE_URL env var is set (Railway PostgreSQL): uses database (persistent)
- Otherwise: falls back to CSV file (local dev)
"""
import os
import pandas as pd
from datetime import datetime
from data_dir import data_path

TRACKER_FILE = data_path('tracker_data.csv')
COLS = ['date', 'player', 'team', 'rating', 'grade', 'projected',
        'line', 'over_odds', 'actual', 'result', 'vs_pitcher']
DATABASE_URL = os.environ.get('DATABASE_URL', '')


def _get_engine():
    if not DATABASE_URL:
        return None
    try:
        from sqlalchemy import create_engine
        url = DATABASE_URL
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql://', 1)
        url = url.replace('postgres://', 'postgresql://', 1)
        if '?' not in url:
            url += '?sslmode=require'
        elif 'sslmode' not in url:
            url += '&sslmode=require'
        return create_engine(url, connect_args={'connect_timeout': 10})
    except Exception:
        return None


def load() -> pd.DataFrame:
    """Load tracker — all columns stored as strings to avoid dtype conflicts."""
    def _clean(df):
        for c in COLS:
            if c not in df.columns:
                df[c] = ''
        df = df[COLS].copy()
        # Convert everything to string and clean up NaN/None
        for c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), '').astype(str)
            df[c] = df[c].replace({'nan': '', 'None': '', 'NaN': ''})
        return df

    engine = _get_engine()
    if engine:
        try:
            df = pd.read_sql('SELECT * FROM tracker ORDER BY date DESC', engine)
            return _clean(df)
        except Exception:
            pass

    if os.path.exists(TRACKER_FILE):
        try:
            df = pd.read_csv(TRACKER_FILE, dtype=str).fillna('')
            return _clean(df)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save(df: pd.DataFrame):
    # Normalize all values to strings before saving
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].astype(str).replace({'nan': '', 'None': '', 'NaN': ''})
    engine = _get_engine()
    if engine:
        try:
            df.to_sql('tracker', engine, if_exists='replace', index=False)
            return
        except Exception:
            pass
    df.to_csv(TRACKER_FILE, index=False)


def recalc_results(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate W/L for any row that has an actual value."""
    df = df.copy()
    df['result'] = df['result'].astype(object)
    for i, row in df.iterrows():
        try:
            actual_str = str(row.get('actual', '')).strip()
            if not actual_str or actual_str == 'nan':
                df.at[i, 'result'] = ''
                continue
            actual = float(actual_str)
            line_val = str(row.get('line', '')).strip()
            line = float(line_val) if line_val and line_val not in ('nan', '') else 1.5
            df.at[i, 'result'] = 'W' if actual > line else 'L'
        except (ValueError, TypeError):
            df.at[i, 'result'] = ''
    return df


def update_rating_if_exists(player_name: str, game_date: str, rating, grade: str,
                            projected, vs_pitcher: str = '') -> bool:
    """Update an existing tracker entry's rating if no actual has been recorded yet.
    Does NOT add new entries — only updates. Returns True if a row was updated."""
    df = load()
    mask = (df['date'] == game_date) & (df['player'] == player_name)
    if not mask.any():
        return False
    idx = df[mask].index[0]
    if str(df.at[idx, 'actual']).strip() not in ('', 'nan'):
        return False  # don't touch completed bets
    df.at[idx, 'rating']     = rating
    df.at[idx, 'grade']      = grade
    df.at[idx, 'projected']  = projected
    df.at[idx, 'vs_pitcher'] = vs_pitcher
    save(df)
    return True


def add_predictions(new_rows: list, game_date: str = None) -> int:
    df    = load()
    today = game_date or datetime.now().strftime('%Y-%m-%d')
    added = 0
    for row in new_rows:
        mask = (df['date'] == today) & (df['player'] == row['player'])
        if mask.any():
            # Update existing pre-game entry if no actual recorded yet
            idx = df[mask].index[0]
            if str(df.at[idx, 'actual']).strip() in ('', 'nan'):
                df.at[idx, 'rating']     = row['rating']
                df.at[idx, 'grade']      = row.get('grade', '')
                df.at[idx, 'projected']  = row['projected']
                df.at[idx, 'vs_pitcher'] = row.get('vs_pitcher', '')
                added += 1
        else:
            df = pd.concat([df, pd.DataFrame([{
                'date':       today,
                'player':     row['player'],
                'team':       row.get('team', ''),
                'rating':     row['rating'],
                'grade':      row.get('grade', ''),
                'projected':  row['projected'],
                'line':       row.get('line', ''),
                'over_odds':  row.get('over_odds', ''),
                'actual':     '',
                'result':     '',
                'vs_pitcher': row.get('vs_pitcher', ''),
            }])], ignore_index=True)
            added += 1
    if added:
        save(df)
    return added
