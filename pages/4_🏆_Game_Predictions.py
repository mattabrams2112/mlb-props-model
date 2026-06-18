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
from datetime import datetime
from eastern_time import today_et, today_str_et

from lineup_fetcher import get_todays_lineups
from pitcher_data import get_pitcher_name
from team_logos import logo_img_tag
from shared_styles import inject_styles
from game_pred_engine import (
    load_preds, save_preds, add_game_pred, get_stored_pred,
    get_adjustments, predict_game_formula, margin_to_confidence,
    fetch_actual_winners, update_game_actuals, SEASON,
)

st.set_page_config(page_title="Game Predictions | MLB Props", page_icon="🏆", layout="wide")
inject_styles()

DATABASE_URL = os.environ.get('DATABASE_URL', '')
PREDS_FILE   = 'game_preds.csv'
COLS = ['date', 'game_id', 'away_team', 'home_team', 'away_pitcher', 'home_pitcher',
        'predicted_winner', 'away_proj', 'home_proj', 'margin', 'confidence',
        'actual_winner', 'result']



# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown('## 🏆 Game Predictions')
st.caption('HRR lineup totals + run differential, defense, bullpen, pitcher rest, and weather adjustments.')

today_str = today_str_et()

col_date, col_refresh, col_fetch = st.columns([2, 1, 1])
with col_date:
    selected_date = st.date_input('Date', value=today_et(),
                                  max_value=today_et(),
                                  label_visibility='collapsed')
with col_refresh:
    if st.button('🔄 Refresh', use_container_width=True):
        st.session_state.pop('gp_games', None)
        st.session_state.pop('gp_rows', None)
        st.rerun()
with col_fetch:
    if st.button('⬇️ Fetch Results', type='primary', use_container_width=True):
        with st.spinner('Fetching results...'):
            n = update_game_actuals()
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

# ── Confidence breakdown ───────────────────────────────────────────────────────

st.markdown('---')
st.markdown('### Accuracy by Confidence Level')

conf_levels = [('Strong', '#22c55e'), ('Moderate', '#3b82f6'), ('Lean', '#eab308'), ('Toss-up', '#475569')]

conf_html = '''<table style="width:100%;border-collapse:collapse;font-family:monospace;">
<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:13px;">
<th style="padding:9px 12px;text-align:left;">Confidence</th>
<th style="padding:9px 12px;text-align:center;">Record</th>
<th style="padding:9px 12px;text-align:center;">Win Rate</th>
<th style="padding:9px 12px;text-align:center;">Games</th>
</tr></thead><tbody>'''

for conf, color in conf_levels:
    sub = decided[decided['confidence'].astype(str) == conf]
    w = int((sub['result'] == 'W').sum())
    l = int((sub['result'] == 'L').sum())
    n = w + l
    wr = round(w / n * 100, 1) if n > 0 else None
    wr_str  = f'{wr}%' if wr is not None else '—'
    rec_str = f'{w}-{l}' if n > 0 else '—'
    wr_color = '#22c55e' if (wr or 0) >= 55 else '#eab308' if (wr or 0) >= 50 else '#ef4444' if n > 0 else '#475569'
    conf_html += f'''<tr style="border-bottom:1px solid #1e293b;">
<td style="padding:9px 12px;"><span style="background:{color}30;color:{color};border-radius:4px;padding:2px 8px;font-weight:700;">{conf}</span></td>
<td style="padding:9px 12px;text-align:center;color:#e0f2fe;font-weight:700;">{rec_str}</td>
<td style="padding:9px 12px;text-align:center;color:{wr_color};font-weight:800;">{wr_str}</td>
<td style="padding:9px 12px;text-align:center;color:#94a3b8;">{n}</td>
</tr>'''

# Total row
conf_html += f'''<tr style="background:#0f172a;border-top:2px solid #38bdf8;">
<td style="padding:10px 12px;color:#38bdf8;font-weight:800;">TOTAL</td>
<td style="padding:10px 12px;text-align:center;color:#e0f2fe;font-weight:800;">{wins}-{losses}</td>
<td style="padding:10px 12px;text-align:center;color:{"#22c55e" if (float(pct.strip("%")) if pct != "—" else 0) >= 55 else "#eab308" if (float(pct.strip("%")) if pct != "—" else 0) >= 50 else "#ef4444" if total > 0 else "#475569"};font-weight:800;">{pct}</td>
<td style="padding:10px 12px;text-align:center;color:#94a3b8;">{total}</td>
</tr></tbody></table>'''

st.markdown(conf_html, unsafe_allow_html=True)

st.markdown('---')

# ── Load games ────────────────────────────────────────────────────────────────

if st.session_state.get('gp_date') != date_str:
    st.session_state.pop('gp_games', None)
    st.session_state.pop('gp_rows',  None)
    st.session_state['gp_date'] = date_str

if 'gp_games' not in st.session_state:
    with st.spinner('Fetching games...'):
        st.session_state['gp_games'] = get_todays_lineups(
            selected_date.strftime('%m/%d/%Y')
        )

games = st.session_state.get('gp_games', [])
if not games:
    st.warning('No games found for this date.')
    st.stop()

# ── HRR availability banner ───────────────────────────────────────────────────

try:
    from team_hrr_store import load_team_hrr as _load_hrr_check
    def _has_hrr(team):
        return (st.session_state.get(f'team_hrr_{date_key}_{team}') is not None
                or _load_hrr_check(date_str, team) is not None)
except Exception:
    def _has_hrr(team):
        return st.session_state.get(f'team_hrr_{date_key}_{team}') is not None

_hrr_count = sum(
    1 for g in games
    if _has_hrr(g.get('away_team')) and _has_hrr(g.get('home_team'))
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
            status   = game.get('status', '')
            game_started     = status not in ('Preview', 'Pre-Game', 'Scheduled', 'Warmup', '')
            lineups_official = game.get('lineups_official', False)
            both_pitchers    = home_p != 'TBD' and away_p != 'TBD'

            # Hold off on predictions until both lineups and pitchers are confirmed
            if not game_started and not (lineups_official and both_pitchers):
                stored = get_stored_pred(gid, date_str)
                if not stored:
                    rows.append({
                        'game_id': gid, 'away_team': away, 'home_team': home,
                        'away_pitcher': away_p, 'home_pitcher': home_p,
                        'predicted_winner': None, 'away_proj': None, 'home_proj': None,
                        'margin': None, 'confidence': None, 'source': 'pending', 'adj': {},
                        'lineups_official': lineups_official,
                    })
                    continue

            # If game has started, use stored prediction — never recalculate
            stored = get_stored_pred(gid, date_str)
            if stored and game_started:
                adj = get_adjustments(home, away, home_pid, away_pid, date_str)
                # Use stored pitcher names — live API drops probablePitcher after first pitch
                stored_away_p = stored.get('away_pitcher', away_p)
                stored_home_p = stored.get('home_pitcher', home_p)
                rows.append({
                    'game_id':          gid,
                    'away_team':        away,
                    'home_team':        home,
                    'away_pitcher':     stored_away_p if stored_away_p and stored_away_p != 'TBD' else away_p,
                    'home_pitcher':     stored_home_p if stored_home_p and stored_home_p != 'TBD' else home_p,
                    'predicted_winner': stored.get('predicted_winner', ''),
                    'away_proj':        float(stored.get('away_proj', 0)),
                    'home_proj':        float(stored.get('home_proj', 0)),
                    'margin':           float(stored.get('margin', 0)),
                    'confidence':       stored.get('confidence', 'Toss-up'),
                    'source':           'stored',
                    'adj':              adj,
                })
                continue

            away_hrr = st.session_state.get(f'team_hrr_{date_key}_{away}')
            home_hrr = st.session_state.get(f'team_hrr_{date_key}_{home}')

            # Fall back to persistent store if not in session state
            if away_hrr is None or home_hrr is None:
                try:
                    from team_hrr_store import load_team_hrr as _load_hrr
                    if away_hrr is None:
                        away_hrr = _load_hrr(date_str, away)
                        if away_hrr is not None:
                            st.session_state[f'team_hrr_{date_key}_{away}'] = away_hrr
                    if home_hrr is None:
                        home_hrr = _load_hrr(date_str, home)
                        if home_hrr is not None:
                            st.session_state[f'team_hrr_{date_key}_{home}'] = home_hrr
                except Exception:
                    pass

            adj = get_adjustments(home, away, home_pid, away_pid, date_str)

            if away_hrr is not None and home_hrr is not None:
                adj_home = round(home_hrr + adj['total_adj'], 1)
                adj_away = round(away_hrr, 1)
                margin   = round(adj_home - adj_away, 1)
                winner   = home if margin >= 0 else away
                away_proj, home_proj = away_hrr, adj_home
                source = 'hrr'
            else:
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
                           'actual_winner': '', 'result': ''}, date_str,
                          game_started=game_started)

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
    away = row['away_team']
    home = row['home_team']

    # ── Pending card — lineups not official yet ───────────────────────────────
    if row.get('source') == 'pending':
        _away_p = row.get('away_pitcher', 'TBD')
        _home_p = row.get('home_pitcher', 'TBD')
        _missing = []
        if _away_p == 'TBD': _missing.append(f'{home} SP')
        if _home_p == 'TBD': _missing.append(f'{away} SP')
        if not row.get('lineups_official'): _missing.append('official lineups')
        _wait_txt = ', '.join(_missing) if _missing else 'official lineups'
        st.markdown(
            f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
            f'padding:14px 18px;margin-bottom:8px;display:flex;align-items:center;gap:12px;">'
            f'{logo_img_tag(away, 28)}'
            f'<span style="color:#38bdf8;font-weight:700;">{away}</span>'
            f'<span style="color:#475569;margin:0 6px;">@</span>'
            f'{logo_img_tag(home, 28)}'
            f'<span style="color:#38bdf8;font-weight:700;">{home}</span>'
            f'<span style="color:#eab308;font-size:12px;margin-left:12px;">'
            f'⏳ Awaiting {_wait_txt}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        continue

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

# ── Day-by-day results tracker ────────────────────────────────────────────────

st.markdown('### Day-by-Day Results')

if decided.empty:
    st.caption('No completed results yet.')
else:
    conf_order = ['Strong', 'Moderate', 'Lean', 'Toss-up']
    conf_colors = {'Strong': '#22c55e', 'Moderate': '#3b82f6', 'Lean': '#eab308', 'Toss-up': '#475569'}

    day_html = '''<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:12px;">
<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:12px;">
<th style="padding:8px 10px;text-align:left;">Date</th>
<th style="padding:8px 10px;text-align:center;">Strong</th>
<th style="padding:8px 10px;text-align:center;">Moderate</th>
<th style="padding:8px 10px;text-align:center;">Lean</th>
<th style="padding:8px 10px;text-align:center;">Toss-up</th>
<th style="padding:8px 10px;text-align:center;">Total</th>
<th style="padding:8px 10px;text-align:center;">Win %</th>
</tr></thead><tbody>'''

    total_by_conf = {c: {'w': 0, 'l': 0} for c in conf_order}

    for date in sorted(decided['date'].astype(str).str[:10].unique(), reverse=True):
        day = decided[decided['date'].astype(str).str[:10] == date]
        dw  = int((day['result'] == 'W').sum())
        dl  = int((day['result'] == 'L').sum())
        dn  = dw + dl
        dwr = round(dw / dn * 100, 1) if dn > 0 else None
        wrc = '#22c55e' if (dwr or 0) >= 55 else '#eab308' if (dwr or 0) >= 50 else '#ef4444' if dn > 0 else '#475569'

        cells = ''
        for c in conf_order:
            sub = day[day['confidence'].astype(str) == c]
            cw  = int((sub['result'] == 'W').sum())
            cl  = int((sub['result'] == 'L').sum())
            total_by_conf[c]['w'] += cw
            total_by_conf[c]['l'] += cl
            cc  = conf_colors[c]
            txt = f'{cw}-{cl}' if (cw + cl) > 0 else '—'
            tc  = '#e0f2fe' if (cw + cl) > 0 else '#334155'
            cells += f'<td style="padding:8px 10px;text-align:center;color:{tc};">{txt}</td>'

        day_html += (
            f'<tr style="border-bottom:1px solid #1e293b;">'
            f'<td style="padding:8px 10px;color:#e0f2fe;">{date}</td>'
            f'{cells}'
            f'<td style="padding:8px 10px;text-align:center;color:#e0f2fe;font-weight:700;">{dw}-{dl}</td>'
            f'<td style="padding:8px 10px;text-align:center;color:{wrc};font-weight:800;">{dwr}%</td>'
            f'</tr>'
        )

    # Totals row
    tot_cells = ''
    for c in conf_order:
        tw = total_by_conf[c]['w']; tl = total_by_conf[c]['l']
        txt = f'{tw}-{tl}' if (tw + tl) > 0 else '—'
        tc  = '#e0f2fe' if (tw + tl) > 0 else '#334155'
        tot_cells += f'<td style="padding:9px 10px;text-align:center;color:{tc};font-weight:700;">{txt}</td>'

    tot_wr = round(wins / total * 100, 1) if total > 0 else None
    tot_wrc = '#22c55e' if (tot_wr or 0) >= 55 else '#eab308' if (tot_wr or 0) >= 50 else '#ef4444' if total > 0 else '#475569'
    day_html += (
        f'<tr style="background:#0f172a;border-top:2px solid #38bdf8;">'
        f'<td style="padding:9px 10px;color:#38bdf8;font-weight:800;">TOTAL</td>'
        f'{tot_cells}'
        f'<td style="padding:9px 10px;text-align:center;color:#e0f2fe;font-weight:800;">{wins}-{losses}</td>'
        f'<td style="padding:9px 10px;text-align:center;color:{tot_wrc};font-weight:800;">{tot_wr}%</td>'
        f'</tr>'
    )
    day_html += '</tbody></table>'
    st.markdown(day_html, unsafe_allow_html=True)

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
