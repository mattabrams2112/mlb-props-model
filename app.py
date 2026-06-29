"""Home Dashboard — today's record, season P&L, and best plays at a glance."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from full_tracker import load_all
from eastern_time import today_str_et
from shared_styles import inject_styles
from team_logos import logo_img_tag
from lineup_fetcher import get_todays_lineups
from pitcher_data import get_pitcher_name
from game_pred_engine import (
    load_preds, add_game_pred, get_stored_pred,
    get_adjustments, predict_game_formula, margin_to_confidence,
)

st.set_page_config(page_title="Dashboard | MLB Props", page_icon="⚾", layout="wide")
inject_styles()

# ── Staking constants (mirrors Daily Results) ─────────────────────────────────
UNIT      = 8.0
_WIN_MULT = 100 / 125

def get_units(rating):
    return 1.0

def play_profit(rating, result):
    u = get_units(rating)
    return round(u * UNIT * _WIN_MULT, 2) if result == 'W' else -(u * UNIT)

def play_units_pl(rating, result):
    u = get_units(rating)
    return round(u * _WIN_MULT, 3) if result == 'W' else -u

def rating_color(r):
    if r >= 90: return '#22c55e'
    if r >= 80: return '#38bdf8'
    if r >= 70: return '#eab308'
    return '#94a3b8'

# ── Load data ─────────────────────────────────────────────────────────────────
today_str = today_str_et()

df_raw = load_all()

st.markdown(
    f'<h1 style="margin-bottom:2px;">⚾ MLB Props Dashboard</h1>'
    f'<p style="color:#64748b;margin-top:0;font-size:0.95rem;">'
    f'{datetime.strptime(today_str, "%Y-%m-%d").strftime("%A, %B %d, %Y")}'
    f'</p>',
    unsafe_allow_html=True
)

if df_raw.empty:
    st.info('No plays logged yet. Open **Game View** to load today\'s lineups and start tracking.')
    st.stop()

df_raw['rating']    = pd.to_numeric(df_raw['rating'],    errors='coerce')
df_raw['projected'] = pd.to_numeric(df_raw['projected'], errors='coerce')
df_raw['actual']    = pd.to_numeric(df_raw['actual'],    errors='coerce')
df_raw['date_str']  = df_raw['date'].astype(str).str[:10]

criteria = df_raw[df_raw['rating'] >= 85]
decided  = criteria[criteria['result'].isin(['W', 'L'])]
pending  = criteria[criteria['result'] == '']

# ── Today slice ───────────────────────────────────────────────────────────────
today_decided = decided[decided['date_str'] == today_str]
today_pending = pending[pending['date_str'] == today_str]
today_w = int((today_decided['result'] == 'W').sum())
today_l = int((today_decided['result'] == 'L').sum())
today_profit = sum(
    play_profit(r['rating'], r['result'])
    for _, r in today_decided.iterrows() if pd.notna(r['rating'])
)

# ── Season totals ─────────────────────────────────────────────────────────────
total_w = int((decided['result'] == 'W').sum())
total_l = int((decided['result'] == 'L').sum())
total_n = len(decided)
total_wr = round(total_w / total_n * 100, 1) if total_n > 0 else None
total_units = sum(
    play_units_pl(r['rating'], r['result'])
    for _, r in decided.iterrows() if pd.notna(r['rating'])
)
total_profit = sum(
    play_profit(r['rating'], r['result'])
    for _, r in decided.iterrows() if pd.notna(r['rating'])
)
total_risked = sum(
    get_units(r['rating']) * UNIT
    for _, r in decided.iterrows() if pd.notna(r['rating'])
)
total_roi = round(total_profit / total_risked * 100, 1) if total_risked > 0 else None

# ── Top metrics row ───────────────────────────────────────────────────────────
st.markdown('---')
c1, c2, c3, c4, c5, c6 = st.columns(6)

if today_w + today_l > 0:
    c1.metric('Today Record', f'{today_w}-{today_l}')
    c2.metric('Today Profit', f'{"+" if today_profit >= 0 else ""}${today_profit:.2f}')
else:
    c1.metric('Today Record', '—')
    c2.metric('Today Profit', '—')

c3.metric('Pending Today', len(today_pending))
c4.metric('Season Record', f'{total_w}-{total_l}' if total_n > 0 else '—')
c5.metric('Season Win Rate', f'{total_wr}%' if total_wr else '—')
c6.metric('Season ROI', f'{total_roi}%' if total_roi else '—')

# ── P&L chart ─────────────────────────────────────────────────────────────────
st.markdown('---')

ch1, ch2 = st.columns([3, 1])
with ch1:
    st.markdown('### Season P&L')
with ch2:
    units_str  = f'+{total_units:.2f}u'  if total_units  >= 0 else f'{total_units:.2f}u'
    profit_str = f'+${total_profit:.2f}' if total_profit >= 0 else f'-${abs(total_profit):.2f}'
    unit_color  = '#22c55e' if total_units  >= 0 else '#ef4444'
    profit_color = '#22c55e' if total_profit >= 0 else '#ef4444'
    st.markdown(
        f'<div style="text-align:right;padding-top:8px;">'
        f'<span style="color:{unit_color};font-size:1.4rem;font-weight:800;">{units_str}</span>'
        f'<span style="color:#475569;font-size:0.85rem;margin-left:10px;">{profit_str} · {total_n} plays</span>'
        f'</div>',
        unsafe_allow_html=True
    )

if not decided.empty:
    agg = []
    for date, grp in decided.sort_values('date_str').groupby('date_str'):
        du = sum(play_units_pl(r['rating'], r['result']) for _, r in grp.iterrows() if pd.notna(r['rating']))
        dp = sum(play_profit(r['rating'], r['result'])   for _, r in grp.iterrows() if pd.notna(r['rating']))
        w  = int((grp['result'] == 'W').sum())
        l  = int((grp['result'] == 'L').sum())
        agg.append({'date': date, 'units': du, 'profit': dp, 'w': w, 'l': l})

    agg_df = pd.DataFrame(agg)
    agg_df['cum_units']  = agg_df['units'].cumsum()
    agg_df['cum_profit'] = agg_df['profit'].cumsum()

    line_color  = '#22c55e' if agg_df['cum_units'].iloc[-1] >= 0 else '#ef4444'
    fill_color  = 'rgba(34,197,94,0.08)' if agg_df['cum_units'].iloc[-1] >= 0 else 'rgba(239,68,68,0.08)'

    hover_texts = [
        f"{r['date']}<br><b>{'+' if r['cum_units'] >= 0 else ''}{r['cum_units']:.2f}u</b>"
        f"<br>{r['w']}-{r['l']} on the day"
        for _, r in agg_df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg_df['date'],
        y=agg_df['cum_units'],
        mode='lines+markers',
        line=dict(color=line_color, width=2.5),
        marker=dict(size=5, color=line_color),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertext=hover_texts,
        hoverinfo='text',
    ))
    fig.add_hline(y=0, line_dash='dot', line_color='#334155', line_width=1)
    fig.update_layout(
        height=240,
        margin=dict(t=5, b=5, l=5, r=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#7dd3fc', size=12),
        xaxis=dict(showgrid=False, showline=False, color='#475569', tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor='#1e293b', zeroline=False,
                   ticksuffix='u', color='#475569', tickfont=dict(size=11)),
        showlegend=False,
        hovermode='closest',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ── Today's plays ─────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown("### Today's Plays")

all_today = pd.concat([today_decided, today_pending]).sort_values('rating', ascending=False) \
            if not (today_decided.empty and today_pending.empty) else pd.DataFrame()

if all_today.empty:
    st.info("No plays logged yet today. Open **Game View** to load lineups.")
else:
    rows_html = ''
    for _, row in all_today.iterrows():
        r      = int(row['rating']) if pd.notna(row['rating']) else 75
        u      = get_units(r)
        result = row.get('result', '')
        rc     = rating_color(r)

        if result == 'W':
            badge = '<span style="background:#14532d;color:#22c55e;padding:3px 10px;border-radius:5px;font-weight:700;font-size:12px;">WIN</span>'
        elif result == 'L':
            badge = '<span style="background:#450a0a;color:#ef4444;padding:3px 10px;border-radius:5px;font-weight:700;font-size:12px;">LOSS</span>'
        else:
            badge = '<span style="background:#1e293b;color:#64748b;padding:3px 10px;border-radius:5px;font-size:12px;">Pending</span>'

        p      = play_profit(r, result) if result in ('W', 'L') else None
        pl_str = (f'<span style="color:{"#22c55e" if p >= 0 else "#ef4444"};font-weight:700;">{"+" if p >= 0 else ""}${p:.2f}</span>'
                  if p is not None else '<span style="color:#334155;">—</span>')

        rows_html += (
            f'<tr style="border-bottom:1px solid #1e293b;">'
            f'<td style="padding:10px 12px;color:#e0f2fe;font-weight:600;">{row.get("player","—")}</td>'
            f'<td style="padding:10px 12px;color:#7dd3fc;">{row.get("team","—")}</td>'
            f'<td style="padding:10px 12px;font-size:16px;font-weight:800;color:{rc};">{r}</td>'
            f'<td style="padding:10px 12px;color:#fbbf24;font-weight:700;">{u}u / ${u*UNIT:.0f}</td>'
            f'<td style="padding:10px 12px;color:#94a3b8;">{row["projected"] if pd.notna(row.get("projected")) else "—"}</td>'
            f'<td style="padding:10px 12px;">{badge}</td>'
            f'<td style="padding:10px 12px;">{pl_str}</td>'
            f'</tr>'
        )

    st.markdown(
        '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">'
        '<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">'
        '<th style="padding:9px 12px;text-align:left;">Player</th>'
        '<th style="padding:9px 12px;text-align:left;">Team</th>'
        '<th style="padding:9px 12px;text-align:left;">Rating</th>'
        '<th style="padding:9px 12px;text-align:left;">Stake</th>'
        '<th style="padding:9px 12px;text-align:left;">Proj</th>'
        '<th style="padding:9px 12px;text-align:left;">Result</th>'
        '<th style="padding:9px 12px;text-align:left;">P/L</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>',
        unsafe_allow_html=True
    )

# ── Last 7 days ───────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown('### Last 7 Days')

if decided.empty:
    st.info('No decided plays yet this season.')
else:
    recent_dates = sorted(decided['date_str'].unique())[-7:]
    recent = decided[decided['date_str'].isin(recent_dates)]

    day_rows = []
    for date, grp in recent.groupby('date_str'):
        w  = int((grp['result'] == 'W').sum())
        l  = int((grp['result'] == 'L').sum())
        dp = sum(play_profit(r['rating'], r['result'])   for _, r in grp.iterrows() if pd.notna(r['rating']))
        du = sum(play_units_pl(r['rating'], r['result']) for _, r in grp.iterrows() if pd.notna(r['rating']))
        wr = round(w / (w + l) * 100, 1) if (w + l) > 0 else 0
        day_rows.append({'date': date, 'w': w, 'l': l, 'wr': wr, 'du': du, 'dp': dp})

    day_rows_sorted = sorted(day_rows, key=lambda x: x['date'], reverse=True)

    rows_html2 = ''
    for row in day_rows_sorted:
        wrc  = '#22c55e' if row['wr'] >= 60 else '#eab308' if row['wr'] >= 55.6 else '#ef4444'
        duc  = '#22c55e' if row['du'] >= 0 else '#ef4444'
        du_s = f'+{row["du"]:.2f}u' if row['du'] >= 0 else f'{row["du"]:.2f}u'
        dp_s = f'+${row["dp"]:.2f}' if row['dp'] >= 0 else f'-${abs(row["dp"]):.2f}'
        rows_html2 += (
            f'<tr style="border-bottom:1px solid #1e293b;">'
            f'<td style="padding:9px 12px;color:#e0f2fe;">{row["date"]}</td>'
            f'<td style="padding:9px 12px;color:#e0f2fe;font-weight:700;">{row["w"]}-{row["l"]}</td>'
            f'<td style="padding:9px 12px;color:{wrc};font-weight:800;">{row["wr"]}%</td>'
            f'<td style="padding:9px 12px;color:{duc};font-weight:700;">{du_s}</td>'
            f'<td style="padding:9px 12px;color:{duc};font-weight:700;">{dp_s}</td>'
            f'</tr>'
        )

    st.markdown(
        '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">'
        '<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:11px;text-transform:uppercase;letter-spacing:0.05em;">'
        '<th style="padding:9px 12px;text-align:left;">Date</th>'
        '<th style="padding:9px 12px;text-align:left;">Record</th>'
        '<th style="padding:9px 12px;text-align:left;">Win %</th>'
        '<th style="padding:9px 12px;text-align:left;">Units</th>'
        '<th style="padding:9px 12px;text-align:left;">Profit</th>'
        f'</tr></thead><tbody>{rows_html2}</tbody></table>',
        unsafe_allow_html=True
    )

# ── ML Game Picks ─────────────────────────────────────────────────────────────
st.markdown('---')

_gp_header_col, _gp_refresh_col = st.columns([5, 1])
with _gp_header_col:
    st.markdown('### ML Game Picks — Today')
    st.caption('Predicted winners based on lineup HRR totals + run differential, bullpen, rest, and home-field.')
with _gp_refresh_col:
    if st.button('🔄 Refresh Picks', use_container_width=True, key='dash_gp_refresh'):
        st.session_state.pop('dash_gp_rows', None)
        st.rerun()

CONF_COLOR = {'Strong': '#22c55e', 'Moderate': '#3b82f6', 'Lean': '#eab308', 'Toss-up': '#475569'}

if 'dash_gp_rows' not in st.session_state:
    with st.spinner('Building game picks...'):
        _gp_games = get_todays_lineups()
        _gp_rows  = []

        for _g in _gp_games:
            _home    = _g.get('home_team', '')
            _away    = _g.get('away_team', '')
            _home_pid = _g.get('home_pitcher_id')
            _away_pid = _g.get('away_pitcher_id')
            _home_p  = get_pitcher_name(_home_pid) if _home_pid else 'TBD'
            _away_p  = get_pitcher_name(_away_pid) if _away_pid else 'TBD'
            _gid     = f'{_away}_{_home}'
            _status  = _g.get('status', '')
            _started = _status not in ('Preview', 'Pre-Game', 'Scheduled', 'Warmup', '')

            _stored = get_stored_pred(_gid, today_str)
            if _stored and _started:
                _gp_rows.append({
                    'game_id':   _gid,
                    'away':      _away,  'home':      _home,
                    'away_p':    _stored.get('away_pitcher', _away_p),
                    'home_p':    _stored.get('home_pitcher', _home_p),
                    'winner':    _stored.get('predicted_winner', ''),
                    'away_proj': float(_stored.get('away_proj', 0) or 0),
                    'home_proj': float(_stored.get('home_proj', 0) or 0),
                    'margin':    float(_stored.get('margin', 0) or 0),
                    'conf':      _stored.get('confidence', 'Toss-up'),
                    'result':    _stored.get('result', ''),
                    'actual':    _stored.get('actual_winner', ''),
                })
                continue

            # Check HRR store then run formula
            _date_key = today_str.replace('-', '')
            _away_hrr = st.session_state.get(f'team_hrr_{_date_key}_{_away}')
            _home_hrr = st.session_state.get(f'team_hrr_{_date_key}_{_home}')
            if _away_hrr is None or _home_hrr is None:
                try:
                    from team_hrr_store import load_team_hrr as _lhrr
                    _away_hrr = _away_hrr or _lhrr(today_str, _away)
                    _home_hrr = _home_hrr or _lhrr(today_str, _home)
                except Exception:
                    pass

            _adj = get_adjustments(_home, _away, _home_pid, _away_pid, today_str)

            if _away_hrr is not None and _home_hrr is not None:
                _hp = round(_home_hrr + _adj['total_adj'], 1)
                _ap = round(_away_hrr, 1)
                _margin = round(_hp - _ap, 1)
                _winner = _home if _margin >= 0 else _away
            else:
                _winner, _ap, _hp, _margin, _adj = predict_game_formula(
                    _home, _away, _home_pid, _away_pid, today_str
                )

            _conf = margin_to_confidence(_margin)
            _row  = {
                'game_id':   _gid,
                'away':      _away,  'home':      _home,
                'away_p':    _away_p, 'home_p':    _home_p,
                'winner':    _winner,
                'away_proj': _ap,    'home_proj': _hp,
                'margin':    _margin, 'conf':      _conf,
                'result':    '', 'actual': '',
            }
            _gp_rows.append(_row)
            add_game_pred({
                'game_id': _gid, 'date': today_str,
                'away_team': _away, 'home_team': _home,
                'away_pitcher': _away_p, 'home_pitcher': _home_p,
                'predicted_winner': _winner,
                'away_proj': _ap, 'home_proj': _hp,
                'margin': _margin, 'confidence': _conf,
                'actual_winner': '', 'result': '',
            }, today_str, game_started=_started)

        st.session_state['dash_gp_rows'] = _gp_rows

_gp_rows = st.session_state.get('dash_gp_rows', [])

if not _gp_rows:
    st.info('No games found for today.')
else:
    # Also show season record for picks
    _preds_all = load_preds()
    _pd        = _preds_all[_preds_all['result'].isin(['W', 'L'])] if not _preds_all.empty else pd.DataFrame()
    _gp_w      = int((_pd['result'] == 'W').sum()) if not _pd.empty else 0
    _gp_l      = int((_pd['result'] == 'L').sum()) if not _pd.empty else 0
    _gp_pct    = f'{_gp_w/(_gp_w+_gp_l):.0%}' if (_gp_w + _gp_l) > 0 else '—'
    st.caption(f'Season picks record: **{_gp_w}-{_gp_l}** ({_gp_pct})')

    # Render as a responsive card grid — 2 per row on wide screens
    _pairs = [_gp_rows[i:i+2] for i in range(0, len(_gp_rows), 2)]
    for _pair in _pairs:
        _cols = st.columns(len(_pair))
        for _ci, (_col, _r) in enumerate(zip(_cols, _pair)):
            with _col:
                _away   = _r['away'];  _home   = _r['home']
                _winner = _r['winner']
                _ap     = _r['away_proj']; _hp = _r['home_proj']
                _margin = _r['margin'];    _conf = _r['conf']
                _cc     = CONF_COLOR.get(_conf, '#475569')

                # Result overlay
                _res_html = ''
                if _r['result'] == 'W':
                    _res_html = f'<div style="color:#22c55e;font-size:11px;font-weight:700;margin-top:4px;">✅ {_r["actual"]} won — CORRECT</div>'
                elif _r['result'] == 'L':
                    _res_html = f'<div style="color:#ef4444;font-size:11px;font-weight:700;margin-top:4px;">❌ {_r["actual"]} won — WRONG</div>'

                _away_bold = 'font-weight:800;font-size:15px;' if _winner == _away else 'font-weight:400;font-size:13px;'
                _home_bold = 'font-weight:800;font-size:15px;' if _winner == _home else 'font-weight:400;font-size:13px;'
                _away_col  = '#e0f2fe' if _winner == _away else '#475569'
                _home_col  = '#e0f2fe' if _winner == _home else '#475569'
                _ap_col    = '#22c55e' if _winner == _away else '#64748b'
                _hp_col    = '#22c55e' if _winner == _home else '#64748b'

                st.markdown(
                    f'<div style="background:#0f1f38;border:1px solid #1e3a5f;border-radius:12px;'
                    f'padding:14px 12px;margin-bottom:10px;">'

                    # Matchup row
                    f'<div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">'

                    # Away team
                    f'<div style="text-align:center;flex:1;">'
                    f'{logo_img_tag(_away, 40)}'
                    f'<div style="color:{_away_col};{_away_bold}margin-top:4px;">{_away}</div>'
                    f'<div style="font-size:10px;color:#475569;margin-top:1px;">{_r["away_p"]}</div>'
                    f'<div style="font-size:20px;font-weight:800;color:{_ap_col};margin-top:3px;">{_ap}</div>'
                    f'<div style="font-size:9px;color:#334155;">proj HRR</div>'
                    f'</div>'

                    # Center
                    f'<div style="text-align:center;flex:0 0 70px;">'
                    f'<div style="font-size:11px;color:#334155;margin-bottom:4px;">@</div>'
                    f'<span style="background:{_cc};color:#000;border-radius:5px;'
                    f'padding:2px 7px;font-size:10px;font-weight:800;">{_conf}</span>'
                    f'<div style="font-size:10px;color:#475569;margin-top:4px;">gap {abs(_margin):.1f}</div>'
                    f'{_res_html}'
                    f'</div>'

                    # Home team
                    f'<div style="text-align:center;flex:1;">'
                    f'{logo_img_tag(_home, 40)}'
                    f'<div style="color:{_home_col};{_home_bold}margin-top:4px;">{_home}</div>'
                    f'<div style="font-size:10px;color:#475569;margin-top:1px;">{_r["home_p"]}</div>'
                    f'<div style="font-size:20px;font-weight:800;color:{_hp_col};margin-top:3px;">{_hp}</div>'
                    f'<div style="font-size:9px;color:#334155;">proj HRR</div>'
                    f'</div>'

                    f'</div>'

                    # Winner banner
                    f'<div style="text-align:center;margin-top:10px;padding-top:8px;'
                    f'border-top:1px solid #1e293b;">'
                    f'<span style="font-size:11px;color:#64748b;">Pick: </span>'
                    f'<span style="font-size:13px;font-weight:800;color:#38bdf8;">{_winner}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
