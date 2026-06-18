"""
Persistent store for team lineup HRR totals.
Written by Game View when a lineup is computed; read by Game Predictions.
"""
import os
import pandas as pd
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL', '')
CACHE_FILE   = 'team_hrr_cache.csv'
COLS         = ['date', 'team', 'hrr_total', 'updated_at']


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


def _load_all() -> pd.DataFrame:
    engine = _get_engine()
    if engine:
        try:
            return pd.read_sql('SELECT * FROM team_hrr_cache', engine)
        except Exception:
            pass
    if os.path.exists(CACHE_FILE):
        try:
            return pd.read_csv(CACHE_FILE, dtype={'date': str, 'team': str})
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def _save_all(df: pd.DataFrame):
    engine = _get_engine()
    if engine:
        try:
            df.to_sql('team_hrr_cache', engine, if_exists='replace', index=False)
            return
        except Exception:
            pass
    df.to_csv(CACHE_FILE, index=False)


def save_team_hrr(date_str: str, team: str, hrr_total: float):
    """Persist a team's lineup HRR total for the given date (YYYY-MM-DD)."""
    df   = _load_all()
    mask = (df['date'] == date_str) & (df['team'] == team)
    if mask.any():
        df.loc[mask, 'hrr_total']  = hrr_total
        df.loc[mask, 'updated_at'] = datetime.now().isoformat()
    else:
        new = pd.DataFrame([{
            'date':       date_str,
            'team':       team,
            'hrr_total':  hrr_total,
            'updated_at': datetime.now().isoformat(),
        }])
        df = pd.concat([df, new], ignore_index=True)
    _save_all(df)


def load_team_hrr(date_str: str, team: str):
    """Return stored HRR total for team on date, or None if not found."""
    df = _load_all()
    if df.empty:
        return None
    match = df[(df['date'] == date_str) & (df['team'] == team)]
    if match.empty:
        return None
    return float(match.iloc[0]['hrr_total'])
