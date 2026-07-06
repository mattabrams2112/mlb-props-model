"""
Shared game prediction engine.
Handles storage, contextual adjustments, formula predictions, and confidence labels.
Imported by both app.py (dashboard) and pages/4_Game_Predictions.py.
"""
import os
import pandas as pd
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL', '')
PREDS_FILE   = 'game_preds.csv'
COLS = ['date', 'game_id', 'away_team', 'home_team', 'away_pitcher', 'home_pitcher',
        'predicted_winner', 'away_proj', 'home_proj', 'margin', 'confidence',
        'actual_winner', 'result']

SEASON = datetime.now().year


# ── Storage ───────────────────────────────────────────────────────────────────

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


def load_preds() -> pd.DataFrame:
    engine = _get_engine()
    if engine:
        try:
            df = pd.read_sql('SELECT * FROM game_predictions ORDER BY date DESC', engine)
            for c in COLS:
                if c not in df.columns:
                    df[c] = ''
            return df[COLS]
        except Exception:
            pass
    if os.path.exists(PREDS_FILE):
        try:
            return pd.read_csv(PREDS_FILE, dtype=str).fillna('')
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save_preds(df: pd.DataFrame):
    engine = _get_engine()
    if engine:
        try:
            df.to_sql('game_predictions', engine, if_exists='replace', index=False)
            return
        except Exception:
            pass
    df.to_csv(PREDS_FILE, index=False)


def add_game_pred(row: dict, game_date: str, game_started: bool = False):
    from eastern_time import today_str_et
    df   = load_preds()
    today = today_str_et()
    match = (not df.empty and
             (df['game_id'].astype(str) == str(row['game_id'])) &
             (df['date'].astype(str).str[:10] == game_date))
    if match.any():
        if not game_started and game_date >= today:
            idx = df[match].index[0]
            for c in ['predicted_winner', 'away_proj', 'home_proj', 'margin',
                      'confidence', 'away_pitcher', 'home_pitcher']:
                if c in row:
                    df.at[idx, c] = row[c]
            save_preds(df)
        return
    new = pd.DataFrame([{c: row.get(c, '') for c in COLS}])
    df  = pd.concat([df, new], ignore_index=True)
    save_preds(df)


def get_stored_pred(game_id: str, game_date: str):
    df = load_preds()
    if df.empty:
        return None
    match = df[(df['game_id'].astype(str) == str(game_id)) &
               (df['date'].astype(str).str[:10] == game_date)]
    return match.iloc[0].to_dict() if not match.empty else None


# ── Contextual adjustments ────────────────────────────────────────────────────

def get_adjustments(home, away, home_pid, away_pid, game_date):
    from team_stats import get_team_recent_scoring, get_team_defense_rating
    from bullpen_data import get_bullpen_stats
    from pitcher_data import get_pitcher_rest_days
    from stadium_weather import get_stadium_weather

    home_sc = get_team_recent_scoring(home)
    away_sc = get_team_recent_scoring(away)

    home_rd  = home_sc.get('team_runs_avg', 4.5) - home_sc.get('team_runs_allowed_avg', 4.5)
    away_rd  = away_sc.get('team_runs_avg', 4.5) - away_sc.get('team_runs_allowed_avg', 4.5)
    form_adj = round((home_rd - away_rd) * 0.20, 2)

    home_def    = get_team_defense_rating(home, SEASON).get('def_rating', 0.0)
    away_def    = get_team_defense_rating(away, SEASON).get('def_rating', 0.0)
    defense_adj = round((away_def - home_def) * 0.15, 2)

    home_bp_era = get_bullpen_stats(home, SEASON).get('bp_era', 4.20)
    away_bp_era = get_bullpen_stats(away, SEASON).get('bp_era', 4.20)
    bp_adj      = round((away_bp_era - home_bp_era) * 0.12, 2)

    home_rest = get_pitcher_rest_days(home_pid, SEASON, game_date).get('rest_factor', 0.0) if home_pid else 0.0
    away_rest = get_pitcher_rest_days(away_pid, SEASON, game_date).get('rest_factor', 0.0) if away_pid else 0.0
    rest_adj  = round((home_rest - away_rest) * 0.15, 2)

    home_field = 0.30

    try:
        wx   = get_stadium_weather(home)
        temp = wx.get('temp_f', 72)
    except Exception:
        temp = 72

    temp_note = ('❄️ {:.0f}°F — cold'.format(temp) if temp < 45 else
                 '🌡️ {:.0f}°F — hot'.format(temp) if temp > 88 else
                 '🌤️ {:.0f}°F'.format(temp))

    total_adj = form_adj + defense_adj + bp_adj + rest_adj + home_field

    return {
        'total_adj':   round(total_adj, 2),
        'form_adj':    form_adj,
        'defense_adj': defense_adj,
        'bp_adj':      bp_adj,
        'rest_adj':    rest_adj,
        'home_field':  home_field,
        'temp':        temp,
        'temp_note':   temp_note,
        'home_rd':     round(home_rd, 2),
        'away_rd':     round(away_rd, 2),
        'home_bp_era': home_bp_era,
        'away_bp_era': away_bp_era,
    }


# ── Formula fallback prediction ────────────────────────────────────────────────

def predict_game_formula(home, away, home_pid, away_pid, game_date):
    from pitcher_data import get_pitcher_season_stats
    from weather import get_park_factor
    from team_stats import get_team_recent_scoring

    base = 12.0
    hp   = get_pitcher_season_stats(home_pid) if home_pid else {}
    ap   = get_pitcher_season_stats(away_pid) if away_pid else {}

    h_era  = hp.get('opp_era', 4.50); h_fip = hp.get('opp_fip', h_era)
    a_era  = ap.get('opp_era', 4.50); a_fip = ap.get('opp_fip', a_era)
    home_pq = 0.55 * h_era + 0.45 * h_fip
    away_pq = 0.55 * a_era + 0.45 * a_fip

    park = get_park_factor(home)
    ht   = get_team_recent_scoring(home)
    at   = get_team_recent_scoring(away)

    away_proj = base * (home_pq / 4.50) * (at.get('team_runs_avg', 4.5) / 4.50) * (ht.get('team_runs_allowed_avg', 4.5) / 4.50) * park
    home_proj = base * (away_pq / 4.50) * (ht.get('team_runs_avg', 4.5) / 4.50) * (at.get('team_runs_allowed_avg', 4.5) / 4.50) * park

    adj        = get_adjustments(home, away, home_pid, away_pid, game_date)
    home_proj += adj['total_adj']

    away_proj = round(min(max(away_proj, 1.5), 15.0), 1)
    home_proj = round(min(max(home_proj, 1.5), 15.0), 1)
    margin    = round(home_proj - away_proj, 1)
    winner    = home if margin >= 0 else away
    return winner, away_proj, home_proj, margin, adj


# ── Confidence label ──────────────────────────────────────────────────────────

def margin_to_confidence(margin):
    a = abs(margin)
    if a >= 4.0:  return 'Strong'
    if a >= 2.0:  return 'Moderate'
    if a >= 0.75: return 'Lean'
    return 'Toss-up'


# ── Fetch actual results ──────────────────────────────────────────────────────

def fetch_actual_winners(game_date: str) -> dict:
    import statsapi
    from lineup_fetcher import TEAM_ABBR
    results = {}
    try:
        date_fmt = datetime.strptime(game_date, '%Y-%m-%d').strftime('%m/%d/%Y')
        for g in statsapi.schedule(date=date_fmt, sportId=1):
            if g.get('status') not in ('Final', 'Game Over', 'Completed Early'):
                continue
            away_s = int(g.get('away_score', 0) or 0)
            home_s = int(g.get('home_score', 0) or 0)
            away_a = TEAM_ABBR.get(g.get('away_name', ''), g.get('away_name', '')[:3].upper())
            home_a = TEAM_ABBR.get(g.get('home_name', ''), g.get('home_name', '')[:3].upper())
            results[f'{away_a}_{home_a}'] = home_a if home_s > away_s else away_a
    except Exception:
        pass
    return results


def update_game_actuals() -> int:
    from eastern_time import today_str_et
    df = load_preds()
    if df.empty:
        return 0
    today   = today_str_et()
    pending = df[df['result'].astype(str).str.strip() == '']
    if pending.empty:
        return 0
    updated = 0
    for game_date in pending['date'].astype(str).str[:10].unique():
        winners = fetch_actual_winners(game_date)
        if not winners:
            continue
        for i in df[df['date'].astype(str).str[:10] == game_date].index:
            if str(df.at[i, 'result']).strip():
                continue
            winner = winners.get(str(df.at[i, 'game_id']))
            if winner:
                df.at[i, 'actual_winner'] = winner
                df.at[i, 'result'] = 'W' if winner == df.at[i, 'predicted_winner'] else 'L'
                updated += 1
    if updated:
        save_preds(df)
    return updated
