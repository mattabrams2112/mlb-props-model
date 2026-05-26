"""
Game Predictions — model-driven winner picks for today's MLB games.
Uses pitcher quality, team offense/defense, and park factors to
project runs for each side and predict the winner.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import statsapi
from datetime import datetime, timedelta

from lineup_fetcher import get_todays_lineups
from pitcher_data import get_pitcher_season_stats, get_pitcher_name
from weather import get_park_factor
from team_stats import get_team_recent_scoring
from team_logos import logo_img_tag

st.set_page_config(page_title="Game Predictions | MLB Props", page_icon="🏆", layout="wide")

st.markdown("""
<style>
  h1,h2,h3{color:#38bdf8!important;}
  .stMarkdown p,label,.stCaption{color:#7dd3fc!important;}
  .stMetric label{color:#38bdf8!important;}
  .stMetric [data-testid="metric-container"]>div{color:#e0f2fe!important;}
  .block-container{padding-top:1rem;}
</style>
""", unsafe_allow_html=True)

DATABASE_URL = os.environ.get('DATABASE_URL', '')
PREDS_FILE   = 'game_preds.csv'
COLS = ['date', 'game_id', 'away_team', 'home_team', 'away_pitcher', 'home_pitcher',
        'predicted_winner', 'away_proj', 'home_proj', 'margin', 'confidence',
        'actual_winner', 'result']


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


def add_game_pred(row: dict, game_date: str):
    df = load_preds()
    exists = (not df.empty and
              (df['game_id'].astype(str) == str(row['game_id'])).any())
    if not exists:
        new = pd.DataFrame([{c: row.get(c, '') for c in COLS}])
        df  = pd.concat([df, new], ignore_index=True)
        save_preds(df)


# ── Prediction model ──────────────────────────────────────────────────────────

def predict_game(home_team, away_team, home_pitcher_id, away_pitcher_id):
    """Project runs for each team and pick a winner."""
    base = 4.50

    hp = get_pitcher_season_stats(home_pitcher_id) if home_pitcher_id else {}
    ap = get_pitcher_season_stats(away_pitcher_id) if away_pitcher_id else {}

    # Blend ERA and FIP for each pitcher
    h_era = hp.get('opp_era', 4.50); h_fip = hp.get('opp_fip', h_era)
    a_era = ap.get('opp_era', 4.50); a_fip = ap.get('opp_fip', a_era)
    home_p_quality = 0.55 * h_era + 0.45 * h_fip
    away_p_quality = 0.55 * a_era + 0.45 * a_fip

    park = get_park_factor(home_team)

    ht = get_team_recent_scoring(home_team)
    at = get_team_recent_scoring(away_team)
    home_off = ht.get('team_runs_avg', 4.5)
    away_off = at.get('team_runs_avg', 4.5)
    home_def = ht.get('team_runs_allowed_avg', 4.5)
    away_def = at.get('team_runs_allowed_avg', 4.5)

    # Projected runs = blend of pitcher quality + team offense + opponent defense + park
    away_proj = (
        base
        * (home_p_quality / 4.50)   # better home pitcher = fewer away runs
        * (away_off / 4.50)          # better away offense = more away runs
        * (home_def / 4.50)          # worse home defense = more away runs
        * park
    )
    home_proj = (
        base
        * (away_p_quality / 4.50)   # better away pitcher = fewer home runs
        * (home_off / 4.50)          # better home offense = more home runs
        * (away_def / 4.50)          # worse away defense = more home runs
        * park
        * 1.02                        # small home field edge
    )

    away_proj = round(min(max(away_proj, 1.5), 12.0), 1)
    home_proj = round(min(max(home_proj, 1.5), 12.0), 1)
    margin    = round(home_proj - away_proj, 1)

    if abs(margin) < 0.3:
        confidence = 'Toss-up'
    elif abs(margin) < 0.7:
        confidence = 'Lean'
    elif abs(margin) < 1.3:
        confidence = 'Moderate'
    else:
        confidence = 'Strong'

    winner = home_team if margin >= 0 else away_team
    return winner, away_proj, home_proj, margin, confidence


# ── Fetch actual results ───────────────────────────────────────────────────────

def fetch_actual_winners(game_date: str) -> dict:
    """Returns {game_id: winner_abbr} for completed games. game_id = away+home."""
    from lineup_fetcher import TEAM_ABBR
    NAME_TO_ABBR = {v: k for k, v in {
        name: abbr for name, abbr in
        [(n, a) for n, a in TEAM_ABBR.items()]
    }.items()}

    results = {}
    try:
        date_fmt = datetime.strptime(game_date, '%Y-%m-%d').strftime('%m/%d/%Y')
        games    = statsapi.schedule(date=date_fmt, sportId=1)
        for g in games:
            if g.get('status') not in ('Final', 'Game Over', 'Completed Early'):
                continue
            away_score = int(g.get('away_score', 0) or 0)
            home_score = int(g.get('home_score', 0) or 0)
            away_name  = g.get('away_name', '')
            home_name  = g.get('home_name', '')
            away_abbr  = NAME_TO_ABBR.get(away_name, away_name[:3].upper())
            home_abbr  = NAME_TO_ABBR.get(home_name, home_name[:3].upper())
            gid        = f'{away_abbr}_{home_abbr}'
            winner     = home_abbr if home_score > away_score else away_abbr
            results[gid] = winner
    except Exception:
        pass
    return results


def update_actuals():
    df    = load_preds()
    if df.empty:
        return 0
    today   = datetime.now().strftime('%Y-%m-%d')
    pending = df[(df['result'].astype(str).str.strip() == '') &
                 (df['date'].astype(str).str[:10] < today)]
    if pending.empty:
        return 0
    updated = 0
    for game_date in pending['date'].astype(str).str[:10].unique():
        winners = fetch_actual_winners(game_date)
        if not winners:
            continue
        date_rows = df[df['date'].astype(str).str[:10] == game_date]
        for i in date_rows.index:
            if str(df.at[i, 'result']).strip():
                continue
            gid    = str(df.at[i, 'game_id'])
            winner = winners.get(gid)
            if winner:
                df.at[i, 'actual_winner'] = winner
                df.at[i, 'result'] = 'W' if winner == df.at[i, 'predicted_winner'] else 'L'
                updated += 1
    if updated:
        save_preds(df)
    return updated


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown('## 🏆 Game Predictions')
st.caption('Winner picks using lineup HRR totals from Game View (most accurate) or pitcher/park formula as fallback.')

today_str = datetime.now().strftime('%Y-%m-%d')

col_date, col_refresh, col_fetch = st.columns([2, 1, 1])
with col_date:
    selected_date = st.date_input('Date', value=datetime.now().date(),
                                  max_value=datetime.now().date(),
                                  label_visibility='collapsed')
with col_refresh:
    if st.button('🔄 Refresh Games', use_container_width=True):
        st.session_state.pop('gp_games', None)
        st.session_state.pop('gp_rows', None)
        st.rerun()
with col_fetch:
    if st.button('⬇️ Fetch Results', type='primary', use_container_width=True):
        with st.spinner('Fetching actual results...'):
            n = update_actuals()
        st.success(f'Updated {n} game(s)!' if n else 'No new results yet.')
        st.rerun()

date_str = selected_date.strftime('%Y-%m-%d')

# ── Record summary ────────────────────────────────────────────────────────────

preds_df = load_preds()
decided  = preds_df[preds_df['result'].isin(['W', 'L'])]
wins     = int((decided['result'] == 'W').sum())
losses   = int((decided['result'] == 'L').sum())
total    = wins + losses
pct      = f"{wins / total:.0%}" if total > 0 else '—'

c1, c2, c3, c4 = st.columns(4)
c1.metric('Record',  f'{wins} - {losses}')
c2.metric('Win %',   pct)
c3.metric('Decided', total)
c4.metric('Pending', int((preds_df['result'].astype(str).str.strip() == '').sum()))

st.markdown('---')

# ── Load games ────────────────────────────────────────────────────────────────

if st.session_state.get('gp_date') != date_str:
    st.session_state.pop('gp_games', None)
    st.session_state.pop('gp_rows',  None)
    st.session_state['gp_date'] = date_str

if 'gp_games' not in st.session_state:
    with st.spinner('Fetching today\'s games...'):
        st.session_state['gp_games'] = get_todays_lineups(
            selected_date.strftime('%m/%d/%Y') if date_str != today_str else None
        )

games = st.session_state.get('gp_games', [])

if not games:
    st.warning('No games found for this date.')
    st.stop()

# ── Build predictions ─────────────────────────────────────────────────────────

date_key = selected_date.strftime('%Y%m%d')

# Check how many games have HRR totals from Game View
_hrr_available = sum(
    1 for g in games
    if st.session_state.get(f'team_hrr_{date_key}_{g.get("away_team")}') is not None
    and st.session_state.get(f'team_hrr_{date_key}_{g.get("home_team")}') is not None
)
if _hrr_available == len(games):
    st.success('✅ Using lineup HRR totals from Game View — most accurate predictions.')
elif _hrr_available > 0:
    st.info(f'⚡ {_hrr_available}/{len(games)} games using lineup HRR totals. Open Game View for the rest.')
else:
    st.warning('⚠️ Load the **Game View** page first for lineup-based predictions. Using pitcher/park formula as fallback.')

if 'gp_rows' not in st.session_state:
    rows = []
    with st.spinner(f'Generating predictions for {len(games)} games...'):
        for game in games:
            home = game.get('home_team', '')
            away = game.get('away_team', '')
            home_pid = game.get('home_pitcher_id')
            away_pid = game.get('away_pitcher_id')
            home_p = get_pitcher_name(home_pid) if home_pid else 'TBD'
            away_p = get_pitcher_name(away_pid) if away_pid else 'TBD'
            gid    = f'{away}_{home}'

            # Use lineup HRR totals from Game View if available
            away_hrr = st.session_state.get(f'team_hrr_{date_key}_{away}')
            home_hrr = st.session_state.get(f'team_hrr_{date_key}_{home}')

            if away_hrr is not None and home_hrr is not None:
                margin = round(home_hrr - away_hrr, 1)
                abs_m  = abs(margin)
                confidence = ('Strong' if abs_m >= 3.0 else
                              'Moderate' if abs_m >= 1.5 else
                              'Lean' if abs_m >= 0.5 else 'Toss-up')
                winner    = home if margin >= 0 else away
                away_proj = away_hrr
                home_proj = home_hrr
            else:
                winner, away_proj, home_proj, margin, confidence = predict_game(
                    home, away, home_pid, away_pid
                )
            rows.append({
                'game_id':          gid,
                'away_team':        away,
                'home_team':        home,
                'away_pitcher':     away_p,
                'home_pitcher':     home_p,
                'predicted_winner': winner,
                'away_proj':        away_proj,
                'home_proj':        home_proj,
                'margin':           margin,
                'confidence':       confidence,
            })

            # Save to tracker
            add_game_pred({**rows[-1], 'date': date_str,
                           'actual_winner': '', 'result': ''}, date_str)

    st.session_state['gp_rows'] = rows

rows = st.session_state.get('gp_rows', [])

# ── Display predictions ───────────────────────────────────────────────────────

CONF_COLOR = {
    'Strong':   '#22c55e',
    'Moderate': '#3b82f6',
    'Lean':     '#eab308',
    'Toss-up':  '#475569',
}

for row in rows:
    away  = row['away_team']
    home  = row['home_team']
    win   = row['predicted_winner']
    ap    = row['away_proj']
    hp    = row['home_proj']
    margin = row['margin']
    conf  = row['confidence']
    cc    = CONF_COLOR.get(conf, '#475569')

    away_bold = 'font-weight:800;font-size:16px;' if win == away else 'color:#94a3b8;'
    home_bold = 'font-weight:800;font-size:16px;' if win == home else 'color:#94a3b8;'
    win_arrow = '← WIN' if win == away else 'WIN →'
    arrow_side = 'left' if win == away else 'right'

    # Check if we have an actual result for this game
    gid_match = preds_df[
        (preds_df['game_id'] == row['game_id']) &
        (preds_df['date'].astype(str).str[:10] == date_str)
    ]
    actual_html = ''
    if not gid_match.empty:
        actual = str(gid_match.iloc[0].get('actual_winner', '')).strip()
        result = str(gid_match.iloc[0].get('result', '')).strip()
        if result == 'W':
            actual_html = f'<span style="color:#22c55e;font-weight:700;">✅ {actual} won — CORRECT</span>'
        elif result == 'L':
            actual_html = f'<span style="color:#ef4444;font-weight:700;">❌ {actual} won — WRONG</span>'

    st.markdown(
        f'<div style="background:#1e293b;border:1px solid #1e40af;border-radius:10px;'
        f'padding:14px 18px;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'

        # Away side
        f'<div style="text-align:center;flex:1;">'
        f'{logo_img_tag(away, 36)}<br>'
        f'<span style="{away_bold}color:{"#e0f2fe" if win==away else "#94a3b8"};">{away}</span><br>'
        f'<span style="font-size:11px;color:#475569;">{row["away_pitcher"]}</span><br>'
        f'<span style="font-size:22px;font-weight:800;color:{"#22c55e" if win==away else "#94a3b8"};">{ap}</span>'
        f'</div>'

        # Center
        f'<div style="text-align:center;flex:1;">'
        f'<div style="font-size:13px;color:#475569;">total proj HRR</div>'
        f'<div style="font-size:20px;color:#475569;margin:4px 0;">@</div>'
        f'<div style="margin-top:6px;">'
        f'<span style="background:{cc};color:#000;border-radius:5px;padding:3px 10px;'
        f'font-size:12px;font-weight:800;">{conf}</span>'
        f'</div>'
        f'<div style="font-size:11px;color:#475569;margin-top:4px;">'
        f'margin: {abs(margin):.1f} runs</div>'
        f'{f"<div style=margin-top:6px;>{actual_html}</div>" if actual_html else ""}'
        f'</div>'

        # Home side
        f'<div style="text-align:center;flex:1;">'
        f'{logo_img_tag(home, 36)}<br>'
        f'<span style="color:{"#e0f2fe" if win==home else "#94a3b8"};">{home}</span><br>'
        f'<span style="font-size:11px;color:#475569;">{row["home_pitcher"]}</span><br>'
        f'<span style="font-size:22px;font-weight:800;color:{"#22c55e" if win==home else "#94a3b8"};">{hp}</span>'
        f'</div>'

        f'</div></div>',
        unsafe_allow_html=True
    )

st.markdown('---')

# ── Historical results ────────────────────────────────────────────────────────

st.markdown('### Past Results')
if decided.empty:
    st.caption('No completed results yet. Click **Fetch Results** after games finish.')
else:
    for date in sorted(decided['date'].astype(str).str[:10].unique(), reverse=True):
        day = decided[decided['date'].astype(str).str[:10] == date]
        dw  = int((day['result'] == 'W').sum())
        dl  = int((day['result'] == 'L').sum())
        dpct = f"{dw/(dw+dl):.0%}" if (dw+dl) > 0 else '—'
        with st.expander(f"**{date}** — {dw}-{dl} ({dpct})", expanded=False):
            disp = day[['away_team','home_team','predicted_winner','away_proj',
                         'home_proj','actual_winner','result']].copy()
            disp['result'] = disp['result'].map({'W': '✅ W', 'L': '❌ L'})
            st.dataframe(disp.rename(columns={
                'away_team':'Away','home_team':'Home',
                'predicted_winner':'Predicted','away_proj':'Away Proj',
                'home_proj':'Home Proj','actual_winner':'Actual','result':'Result'
            }), hide_index=True, use_container_width=True)
