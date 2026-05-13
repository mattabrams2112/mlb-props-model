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
        return create_engine(url)
    except Exception:
        return None


def load() -> pd.DataFrame:
    engine = _get_engine()
    if engine:
        try:
            df = pd.read_sql('SELECT * FROM tracker ORDER BY date DESC', engine)
            for c in COLS:
                if c not in df.columns:
                    df[c] = ''
            return df[COLS]
        except Exception:
            pass

    if os.path.exists(TRACKER_FILE):
        try:
            df = pd.read_csv(TRACKER_FILE)
            for c in COLS:
                if c not in df.columns:
                    df[c] = ''
            return df[COLS]
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save(df: pd.DataFrame):
    engine = _get_engine()
    if engine:
        try:
            df.to_sql('tracker', engine, if_exists='replace', index=False)
            return
        except Exception:
            pass
    df.to_csv(TRACKER_FILE, index=False)


def recalc_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i, row in df.iterrows():
        try:
            actual = float(row['actual'])
            line   = float(row['line'])
            df.at[i, 'result'] = 'W' if actual > line else 'L'
        except (ValueError, TypeError):
            df.at[i, 'result'] = ''
    return df


def add_predictions(new_rows: list) -> int:
    df    = load()
    today = datetime.now().strftime('%Y-%m-%d')
    existing = set(zip(df['date'], df['player']))
    added = 0
    for row in new_rows:
        if (today, row['player']) not in existing:
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
