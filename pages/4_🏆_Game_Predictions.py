"""
Game Predictions — model-driven winner picks for today's MLB games.
Primary: lineup HRR totals from Game View + contextual adjustments.
Fallback: pitcher/park/team formula when Game View hasn't been loaded.

Adjustment factors:
  - Recent run differential (team form)
  - Team defense rating
  - Bullpen ERA comparison
  - Starting pitcher rest days
  - Home field (real splits)
  - Weather / temperature
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import statsapi
from datetime import datetime

from lineup_fetcher import get_todays_lineups
from pitcher_data import get_pitcher_season_stats, get_pitcher_name, get_pitcher_rest_days
from weather import get_park_factor
from team_stats import get_team_recent_scoring, get_team_defense_rating
from bullpen_data import get_bullpen_stats
from stadium_weather import get_stadium_weather
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


def add_game_pred(row: dict, game_date: str):
    df = load_preds()
    exists = (not df.empty and
              (df['game_id'].astype(str) == str(row['game_id'])).any())
    if not exists:
        new = pd.DataFrame([{c: row.get(c, '') for c in COLS}])
        df  = pd.concat([df, new], ignore_index=True)
        save_preds(df)


# ── Contextual adjustments ────────────────────────────────────────────────────

def get_adjustments(home, away, home_pid, away_pid, game_date):
    """
    Returns home team advantage score and factor breakdown.
    Positive = home favored, negative = away favored.
    All adjustments are in HRR units.
    """
    home_sc = get_team_recent_scoring(home)
    away_sc = get_team_recent_scoring(away)

    # Recent run differential — how dominant is each team lately
    home_rd = home_sc.get('team_runs_avg', 4.5) - home_sc.get('team_runs_allowed_avg', 4.5)
    away_rd = away_sc.get('team_runs_avg', 4.5) - away_sc.get('team_runs_allowed_avg', 4.5)
    form_adj = round((home_rd - away_rd) * 0.20, 2)

    # Defense — bad defense = more runs allowed (good for opposing offense)
    home_def = get_team_defense_rating(home, SEASON).get('def_rating', 0.0)
    away_def = get_team_defense_rating(away, SEASON).get('def_rating', 0.0)
    # away's bad defense helps home offense, home's bad defense helps away offense
    defense_adj = round((away_def - home_def) * 0.15, 2)

    # Bullpen — lower ERA = stronger bullpen = fewer late-game runs allowed
    home_bp_era = get_bullpen_stats(home, SEASON).get('bp_era', 4.20)
    away_bp_era = get_bullpen_stats(away, SEASON).get('bp_era', 4.20)
    bp_adj = round((away_bp_era - home_bp_era) * 0.12, 2)

    # Pitcher rest — well-rested starter performs better = fewer runs allowed
    home_rest = get_pitcher_rest_days(home_pid, SEASON, game_date).get('rest_factor', 0.0) if home_pid else 0.0
    away_rest = get_pitcher_rest_days(away_pid, SEASON, game_date).get('rest_factor', 0.0) if away_pid else 0.0
    rest_adj = round((home_rest - away_rest) * 0.15, 2)

    # Home field advantage
    home_field = 0.30

    # Weather — temperature affects offense
    try:
        wx = get_stadium_weather(home)
        temp = wx.get('temp_f', 72)
    except Exception:
        temp = 72
    # Cold weather reduces scoring — penalise both teams slightly (affects total HRR)
    # We just note it rather than adjust advantage since it affects both teams
    temp_note = (f'❄️ {temp:.0f}°F — cold, expect lower scoring'  if temp < 45 else
                 f'🌡️ {temp:.0f}°F — hot, hitter-friendly'        if temp > 88 else
                 f'🌤️ {temp:.0f}°F')

    total_adj = form_adj + defense_adj + bp_adj + rest_adj + home_field

    return {
        'total_adj':    round(total_adj, 2),
        'form_adj':     form_adj,
        'defense_adj':  defense_adj,
        'bp_adj':       bp_adj,
        'rest_adj':     rest_adj,
        'home_field':   home_field,
        'temp':         temp,
        'temp_note':    temp_note,
        'home_rd':      round(home_rd, 2),
        'away_rd':      round(away_rd, 2),
        'home_bp_era':  home_bp_era,
        'away_bp_era':  away_bp_era,
    }


# ── Fallback prediction (no HRR totals) ──────────────────────────────────────

def predict_game_formula(home, away, home_pid, away_pid, game_date):
    """Full formula prediction when Game View HRR totals aren't available."""
    base = 4.50
    hp   = get_pitcher_season_stats(home_pid) if home_pid else {}
    ap   = get_pitcher_season_stats(away_pid) if away_pid else {}

    h_era = hp.get('opp_era', 4.50); h_fip = hp.get('opp_fip', h_era)
    a_era = ap.get('opp_era', 4.50); a_fip = ap.get('opp_fip', a_era)
    home_pq = 0.55 * h_era + 0.45 * h_fip
    away_pq = 0.55 * a_era + 0.45 * a_fip

    park    = get_park_factor(home)
    ht      = get_team_recent_scoring(home)
    at      = get_team_recent_scoring(away)

    away_proj = base * (home_pq / 4.50) * (at.get('team_runs_avg', 4.5) / 4.50) * (ht.get('team_runs_allowed_avg', 4.5) / 4.50) * park
    home_proj = base * (away_pq / 4.50) * (ht.get('team_runs_avg', 4.5) / 4.50) * (at.get('team_runs_allowed_avg', 4.5) / 4.50) * park

    # Get adjustments
    adj = get_adjustments(home, away, home_pid, away_pid, game_date)
    home_proj += adj['total_adj']

    away_proj = round(min(max(away_proj, 1.5), 15.0), 1)
    home_proj = round(min(max(home_proj, 1.5), 15.0), 1)
    margin    = round(home_proj - away_proj, 1)
    winner    = home if margin >= 0 else away
    return winner, away_proj, home_proj, margin, adj


# ── Confidence label ──────────────────────────────────────────────────────────

def margin_to_confidence(margin):
    a = abs(margin)
    if a >= 4.0:   return 'Strong'
    if a >= 2.0:   return 'Moderate'
    if a >= 0.75:  return 'Lean'
    return 'Toss-up'


# ── Fetch actual results ───────────────────────────────────────────────────────

def fetch_actual_winners(game_date: str) -> dict:
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


def update_actuals():
    df = load_preds()
    if df.empty:
        return 0
    today   = datetime.now().strftime('%Y-%m-%d')
    # Include today — completed games should be fetched even on same day
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


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown('## 🏆 Game Predictions')
st.caption('HRR lineup totals + run differential, defense, bullpen, pitcher rest, and weather adjustments.')

today_str = datetime.now().strftime('%Y-%m-%d')

col_date, col_refresh, col_fetch = st.columns([2, 1, 1])
with col_date:
    selected_date = st.date_input('Date', value=datetime.now().date(),
                                  max_value=datetime.now().date(),
                                  label_visibility='collapsed')
with col_refresh:
    if st.button('🔄 Refresh', use_container_width=True):
        st.session_state.pop('gp_games', None)
        st.session_state.pop('gp_rows', None)
        st.rerun()
with col_fetch:
    if st.button('⬇️ Fetch Results', type='primary', use_container_width=True):
        with st.spinner('Fetching results...'):
            n = update_actuals()
        st.success(f'Updated {n} game(s)!' if n else 'No new results yet.')
        st.rerun()

date_str = selected_date.strftime('%Y-%m-%d')
date_key = selected_date.strftime('%Y%m%d')

# ── Record summary ────────────────────────────────────────────────────────────

preds_df = load_preds()
decided  = preds_df[preds_df['result'].isin(['W', 'L'])]
wins     = int((decided['result'] == 'W').sum())
losses   = int((decided['result'] == 'L').sum())
total    = wins + losses
pct      = f"{wins/total:.0%}" if total > 0 else '—'

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
    with st.spinner('Fetching games...'):
        st.session_state['gp_games'] = get_todays_lineups(
            selected_date.strftime('%m/%d/%Y') if date_str != today_str else None
        )

games = st.session_state.get('gp_games', [])
if not games:
    st.warning('No games found for this date.')
    st.stop()

# ── HRR availability banner ───────────────────────────────────────────────────

_hrr_count = sum(
    1 for g in games
    if st.session_state.get(f'team_hrr_{date_key}_{g.get("away_team")}') is not None
    and st.session_state.get(f'team_hrr_{date_key}_{g.get("home_team")}') is not None
)
if _hrr_count == len(games):
    st.success('✅ Using lineup HRR totals from Game View + contextual adjustments.')
elif _hrr_count > 0:
    st.info(f'⚡ {_hrr_count}/{len(games)} games using HRR totals. Open Game View for full accuracy.')
else:
    st.warning('⚠️ Open **Game View** first for lineup-based predictions. Using formula fallback.')

# ── Build predictions ─────────────────────────────────────────────────────────

if 'gp_rows' not in st.session_state:
    rows = []
    with st.spinner(f'Building predictions for {len(games)} games...'):
        for game in games:
            home     = game.get('home_team', '')
            away     = game.get('away_team', '')
            home_pid = game.get('home_pitcher_id')
            away_pid = game.get('away_pitcher_id')
            home_p   = get_pitcher_name(home_pid) if home_pid else 'TBD'
            away_p   = get_pitcher_name(away_pid) if away_pid else 'TBD'
            gid      = f'{away}_{home}'

            away_hrr = st.session_state.get(f'team_hrr_{date_key}_{away}')
            home_hrr = st.session_state.get(f'team_hrr_{date_key}_{home}')

            adj = get_adjustments(home, away, home_pid, away_pid, date_str)

            if away_hrr is not None and home_hrr is not None:
                # HRR totals available — apply adjustments on top
                adj_home = round(home_hrr + adj['total_adj'], 1)
                adj_away = round(away_hrr, 1)
                margin   = round(adj_home - adj_away, 1)
                winner   = home if margin >= 0 else away
                away_proj, home_proj = away_hrr, adj_home
                source = 'hrr'
            else:
                # Fallback formula
                winner, away_proj, home_proj, margin, adj = predict_game_formula(
                    home, away, home_pid, away_pid, date_str
                )
                source = 'formula'

            confidence = margin_to_confidence(margin)

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
                'source':           source,
                'adj':              adj,
            })

            add_game_pred({**rows[-1], 'date': date_str,
                           'actual_winner': '', 'result': ''}, date_str)

    st.session_state['gp_rows'] = rows

rows = st.session_state.get('gp_rows', [])

# ── Display ───────────────────────────────────────────────────────────────────

CONF_COLOR = {'Strong': '#22c55e', 'Moderate': '#3b82f6',
              'Lean': '#eab308', 'Toss-up': '#475569'}


def factor_badge(label, value, good_positive=True):
    """Small badge showing a factor adjustment."""
    if abs(value) < 0.05:
        return ''
    color = '#22c55e' if (value > 0) == good_positive else '#ef4444'
    sign  = '+' if value > 0 else ''
    return (f'<span style="background:{color}20;color:{color};border:1px solid {color}40;'
            f'border-radius:4px;padding:1px 6px;font-size:10px;margin:1px;">'
            f'{label} {sign}{value:+.1f}</span>')


for row in rows:
    away   = row['away_team']
    home   = row['home_team']
    win    = row['predicted_winner']
    ap     = row['away_proj']
    hp     = row['home_proj']
    margin = row['margin']
    conf   = row['confidence']
    adj    = row['adj']
    cc     = CONF_COLOR.get(conf, '#475569')

    # Result lookup
    gid_match = preds_df[
        (preds_df['game_id'] == row['game_id']) &
        (preds_df['date'].astype(str).str[:10] == date_str)
    ]
    actual_html = ''
    if not gid_match.empty:
        actual = str(gid_match.iloc[0].get('actual_winner', '')).strip()
        result = str(gid_match.iloc[0].get('result', '')).strip()
        if result == 'W':
            actual_html = f'<div style="color:#22c55e;font-weight:700;margin-top:4px;">✅ {actual} won — CORRECT</div>'
        elif result == 'L':
            actual_html = f'<div style="color:#ef4444;font-weight:700;margin-top:4px;">❌ {actual} won — WRONG</div>'

    # Factor badges
    badges = ''.join([
        factor_badge('Form', adj['form_adj']),
        factor_badge('Defense', adj['defense_adj']),
        factor_badge('Bullpen', adj['bp_adj']),
        factor_badge('Rest', adj['rest_adj']),
    ])

    st.markdown(
        f'<div style="background:#1e293b;border:1px solid #1e40af;border-radius:10px;'
        f'padding:14px 18px;margin-bottom:12px;">'

        # Game card header
        f'<div style="display:flex;align-items:center;justify-content:space-between;">'

        # Away
        f'<div style="text-align:center;flex:1;">'
        f'{logo_img_tag(away, 36)}<br>'
        f'<span style="color:{"#e0f2fe" if win==away else "#94a3b8"};font-weight:{"800" if win==away else "400"};font-size:{"16px" if win==away else "14px"};">{away}</span><br>'
        f'<span style="font-size:10px;color:#475569;">{row["away_pitcher"]}</span><br>'
        f'<span style="font-size:22px;font-weight:800;color:{"#22c55e" if win==away else "#94a3b8"};">{ap}</span><br>'
        f'<span style="font-size:10px;color:#475569;">BP ERA {adj["away_bp_era"]:.2f} · RD {adj["away_rd"]:+.1f}</span>'
        f'</div>'

        # Center
        f'<div style="text-align:center;flex:1;">'
        f'<div style="font-size:11px;color:#475569;">total proj HRR</div>'
        f'<div style="font-size:18px;color:#475569;margin:2px 0;">@</div>'
        f'<span style="background:{cc};color:#000;border-radius:5px;padding:3px 10px;'
        f'font-size:12px;font-weight:800;">{conf}</span>'
        f'<div style="font-size:11px;color:#475569;margin-top:3px;">margin {abs(margin):.1f}</div>'
        f'<div style="font-size:10px;color:#94a3b8;margin-top:2px;">{adj["temp_note"]}</div>'
        f'{actual_html}'
        f'</div>'

        # Home
        f'<div style="text-align:center;flex:1;">'
        f'{logo_img_tag(home, 36)}<br>'
        f'<span style="color:{"#e0f2fe" if win==home else "#94a3b8"};font-weight:{"800" if win==home else "400"};font-size:{"16px" if win==home else "14px"};">{home}</span><br>'
        f'<span style="font-size:10px;color:#475569;">{row["home_pitcher"]}</span><br>'
        f'<span style="font-size:22px;font-weight:800;color:{"#22c55e" if win==home else "#94a3b8"};">{hp}</span><br>'
        f'<span style="font-size:10px;color:#475569;">BP ERA {adj["home_bp_era"]:.2f} · RD {adj["home_rd"]:+.1f}</span>'
        f'</div>'

        f'</div>'

        # Factor badges row
        + (f'<div style="margin-top:8px;border-top:1px solid #1e40af;padding-top:6px;">'
           f'<span style="font-size:10px;color:#475569;">Adjustments: </span>{badges}'
           f'</div>' if badges else '')

        + f'</div>',
        unsafe_allow_html=True
    )

st.markdown('---')

# ── Past results ──────────────────────────────────────────────────────────────

st.markdown('### Past Results')
if decided.empty:
    st.caption('No completed results yet. Click **Fetch Results** after games finish.')
else:
    for date in sorted(decided['date'].astype(str).str[:10].unique(), reverse=True):
        day  = decided[decided['date'].astype(str).str[:10] == date]
        dw   = int((day['result'] == 'W').sum())
        dl   = int((day['result'] == 'L').sum())
        dpct = f"{dw/(dw+dl):.0%}" if (dw+dl) > 0 else '—'
        with st.expander(f"**{date}** — {dw}-{dl} ({dpct})", expanded=False):
            disp = day[['away_team','home_team','predicted_winner','away_proj',
                        'home_proj','actual_winner','result']].copy()
            disp['result'] = disp['result'].map({'W': '✅ W', 'L': '❌ L'})
            st.dataframe(disp.rename(columns={
                'away_team':'Away','home_team':'Home',
                'predicted_winner':'Predicted','away_proj':'Away HRR',
                'home_proj':'Home HRR','actual_winner':'Actual','result':'Result'
            }), hide_index=True, use_container_width=True)
