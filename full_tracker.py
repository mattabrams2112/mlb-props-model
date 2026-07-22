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
from eastern_time import today_str_et

LOG_FILE     = data_path('full_play_log.csv')
DATABASE_URL = os.environ.get('DATABASE_URL', '')
COLS = ['date', 'player', 'team', 'rating', 'grade', 'projected', 'base_proj',
        'line', 'over_odds', 'actual', 'result', 'vs_pitcher', 'is_home', 'pitcher_throws',
        'r30g']   # r30g = player's live 30-game HRR baseline at play time (clean,
                  # no leakage); boom_delta = projected - r30g is derived at read time


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
             pitcher_throws: str = '', r30g: float = None):
    """Log a play. Updates rating/projection only if game hasn't started yet.
    r30g = the player's live 30-game HRR baseline (for boom_delta analysis)."""
    df    = load_all()
    today = game_date or today_str_et()
    # Key on date + player + vs_pitcher so doubleheader games each get a row
    _vp = str(vs_pitcher).strip()
    existing = (not df.empty and
                ((df['date'].astype(str) == today) & (df['player'] == player) &
                 (df['vs_pitcher'].astype(str).str.strip() == _vp)))
    if existing.any():
        idx = df[existing].index[0]
        # Only update if today (ET), game hasn't started yet, and no actual recorded
        is_today = (today == today_str_et())
        if is_today and not game_started and str(df.at[idx, 'actual']).strip() in ('', 'nan'):
            df.at[idx, 'rating']     = rating
            df.at[idx, 'grade']      = grade
            df.at[idx, 'projected']  = projected
            df.at[idx, 'base_proj']  = base_proj if base_proj is not None else ''
            df.at[idx, 'vs_pitcher'] = vs_pitcher
            # Backfill the baseline whenever it's provided and still missing
            if r30g is not None and str(df.at[idx, 'r30g']).strip() in ('', 'nan'):
                df.at[idx, 'r30g'] = r30g
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
        'r30g':           r30g if r30g is not None else '',
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

    A row needs work when it's a past day AND either:
      - its actual hasn't been fetched yet, OR
      - it has an actual but no W/L result and a line now exists (re-grade).
    This means a play that had no line when its actual came in will get
    graded on the next fetch once a line is entered, instead of being
    stuck in "pending" forever.
    """
    df = load_all()
    if df.empty:
        return 0

    today   = today_str_et()   # ET — server runs UTC, don't use datetime.now()
    updated = 0

    df = df.copy()
    df['_d'] = df['date'].astype(str).str[:10]
    _actual = df['actual'].astype(str).str.strip()
    _result = df['result'].astype(str).str.strip()
    _line   = (df['line'].astype(str).str.strip()
               if 'line' in df.columns else pd.Series('', index=df.index))

    no_actual   = _actual.isin(['', 'nan'])
    ungraded    = _result.isin(['', 'nan']) & ~_line.isin(['', 'nan'])
    needs = df[(df['_d'] < today) & (no_actual | ungraded)]
    if needs.empty:
        return 0

    changed = False
    for game_date in needs['_d'].unique():
        date_rows = df[df['_d'] == game_date]
        # Only hit the boxscore API if some row on this date still needs an actual
        need_fetch   = date_rows['actual'].astype(str).str.strip().isin(['', 'nan']).any()
        player_stats = _get_boxscore_stats_for_date(game_date) if need_fetch else {}

        for i in date_rows.index:
            row = df.loc[i]
            has_actual = str(row.get('actual', '')).strip() not in ('', 'nan')

            # 1) Fetch the actual if we don't have it yet
            if not has_actual:
                if not player_stats:
                    continue
                player_lower = str(row.get('player', '')).lower().strip()
                hrr = player_stats.get(player_lower)
                if hrr is None:  # partial last-name fallback
                    parts = player_lower.split()
                    last  = parts[-1] if parts else ''
                    for k, v in player_stats.items():
                        if last and last in k:
                            hrr = v
                            break
                if hrr is None:
                    continue
                df.at[i, 'actual'] = str(hrr)
                updated += 1
                changed = True

            # 2) Grade W/L when we have an actual, a real line, and no result yet
            actual_val = str(df.at[i, 'actual']).strip()
            result_val = str(df.at[i, 'result']).strip()
            line_val   = str(row.get('line', '')).strip()
            if (actual_val not in ('', 'nan') and result_val in ('', 'nan')
                    and line_val and line_val not in ('nan', '') and game_date < today):
                try:
                    df.at[i, 'result'] = 'W' if float(actual_val) > float(line_val) else 'L'
                    changed = True
                except ValueError:
                    pass

    df.drop(columns=['_d'], inplace=True, errors='ignore')
    if changed:
        save_all(df)
    return updated
