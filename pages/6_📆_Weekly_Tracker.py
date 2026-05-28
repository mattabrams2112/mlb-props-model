"""
Weekly Tracker — rating band × projection breakdown per week.
One card per week, newest first. Matches Analytics table style.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from full_tracker import load_all

st.set_page_config(page_title="Weekly Tracker | MLB Props", page_icon="📆", layout="wide")

st.markdown("""
<style>
  h1,h2,h3{color:#38bdf8!important;}
  .stMarkdown p,label,.stCaption{color:#7dd3fc!important;}
  .stMetric label{color:#38bdf8!important;}
  .stMetric [data-testid="metric-container"]>div{color:#e0f2fe!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('## 📆 Weekly Tracker')
st.caption('Rating band × projection breakdown — one card per week, newest first.')

if st.button('🔄 Refresh', use_container_width=False):
    st.rerun()

# ── Load & prep ───────────────────────────────────────────────────────────────

df = load_all()
if df.empty:
    st.info('No play data yet. Open Game View to start logging plays.')
    st.stop()

df['rating']    = pd.to_numeric(df['rating'],    errors='coerce')
df['projected'] = pd.to_numeric(df['projected'], errors='coerce')
df['actual']    = pd.to_numeric(df['actual'],    errors='coerce')
df['date_str']  = df['date'].astype(str).str[:10]

today_str = datetime.now().strftime('%Y-%m-%d')
df = df.copy()
df.loc[df['date_str'] >= today_str, ['actual', 'result']] = ''

# ── Helpers ───────────────────────────────────────────────────────────────────

RATING_BUCKETS = [
    (60, 65, '60-64'), (65, 70, '65-69'), (70, 75, '70-74'),
    (75, 80, '75-79'), (80, 85, '80-84'), (85, 90, '85-89'),
    (90, 95, '90-94'), (95, 101, '95+'),
]
PROJ_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]


def record(sub):
    d = sub[sub['result'].isin(['W', 'L'])]
    if len(d) == 0:
        return None, 0, 0, 0
    w = int((d['result'] == 'W').sum())
    l = int((d['result'] == 'L').sum())
    return round(w / len(d) * 100, 1), len(d), w, l


def fmt(sub):
    wr, n, w, l = record(sub)
    if n == 0:
        return '—'
    return f'{wr}% ({w}-{l})'


def week_start(date_str):
    d = datetime.strptime(date_str, '%Y-%m-%d')
    return (d - timedelta(days=d.weekday())).strftime('%Y-%m-%d')


def week_label(monday_str):
    mon = datetime.strptime(monday_str, '%Y-%m-%d')
    sun = mon + timedelta(days=6)
    return f"Week of {mon.strftime('%b %-d')} – {sun.strftime('%b %-d, %Y')}"


def build_table(data_df) -> str:
    """Return HTML for the rating band × projection table."""
    cols = ['Rating Band', 'Total Plays', 'Total %'] + [f'Proj ≥{pt}' for pt in PROJ_THRESHOLDS]

    html = (
        '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">'
        '<thead><tr style="background:#1e3a5f;">'
    )
    for c in cols:
        align = 'left' if c == 'Rating Band' else 'center'
        html += (
            f'<th style="padding:9px 12px;text-align:{align};color:#38bdf8;'
            f'font-size:12px;font-weight:700;border-bottom:1px solid #1e40af;">{c}</th>'
        )
    html += '</tr></thead><tbody>'

    for lo, hi, label in RATING_BUCKETS:
        band    = data_df[(data_df['rating'] >= lo) & (data_df['rating'] < hi)]
        decided = band[band['result'].isin(['W', 'L'])]
        total_n = len(decided)

        row_vals = {
            'Rating Band': label,
            'Total Plays': str(total_n) if total_n > 0 else '—',
            'Total %':     fmt(band),
        }
        for pt in PROJ_THRESHOLDS:
            row_vals[f'Proj ≥{pt}'] = fmt(band[band['projected'] >= pt])

        html += '<tr style="border-bottom:1px solid #1e293b;">'
        for c in cols:
            val   = row_vals[c]
            align = 'left' if c == 'Rating Band' else 'center'
            color = '#e0f2fe'
            bold  = ''
            if c not in ('Rating Band', 'Total Plays') and val != '—':
                try:
                    pct   = float(val.split('%')[0])
                    color = '#22c55e' if pct >= 60 else '#eab308' if pct >= 52.4 else '#ef4444'
                    bold  = 'font-weight:700;'
                except Exception:
                    pass
            elif c == 'Total Plays':
                color = '#94a3b8'

            html += (
                f'<td style="padding:9px 12px;text-align:{align};color:{color};{bold}">'
                f'{val}</td>'
            )
        html += '</tr>'

    html += '</tbody></table>'
    return html


# ── Week grouping ─────────────────────────────────────────────────────────────

df['week'] = df['date_str'].apply(week_start)
weeks      = sorted(df['week'].dropna().unique(), reverse=True)

if not weeks:
    st.info('No plays with valid dates found.')
    st.stop()

# ── Season totals card ────────────────────────────────────────────────────────

st.markdown('### 📊 Season Totals')
_, s_n, s_w, s_l = record(df)
s_wr = f'{round(s_w/s_n*100,1)}%' if s_n > 0 else '—'

st.markdown(
    f'<div style="background:#1e293b;border:1px solid #1e40af;border-radius:10px;'
    f'padding:14px 18px;margin-bottom:20px;">'
    f'<div style="color:#38bdf8;font-weight:700;font-size:15px;margin-bottom:10px;">'
    f'All Weeks — {s_w}-{s_l} ({s_wr}) · {len(weeks)} weeks tracked</div>'
    + build_table(df) +
    '</div>',
    unsafe_allow_html=True
)

st.markdown('---')
st.markdown('### Weekly Breakdown')

# ── One card per week ─────────────────────────────────────────────────────────

for week_mon in weeks:
    wk_df  = df[df['week'] == week_mon]
    _, wk_n, wk_w, wk_l = record(wk_df)
    wk_wr  = f'{round(wk_w/wk_n*100,1)}%' if wk_n > 0 else '—'
    label  = week_label(week_mon)

    # Day strip
    days     = sorted(wk_df['date_str'].unique())
    day_html = '<div style="display:flex;gap:6px;margin-bottom:10px;">'
    for day in days:
        day_sub = wk_df[wk_df['date_str'] == day]
        _, dn, dw, dl = record(day_sub)
        d_label = datetime.strptime(day, '%Y-%m-%d').strftime('%a %-d')
        d_color = ('#22c55e' if dn > 0 and dw/dn >= 0.60 else
                   '#eab308' if dn > 0 and dw/dn >= 0.524 else
                   '#ef4444' if dn > 0 else '#475569')
        day_html += (
            f'<div style="text-align:center;padding:5px 10px;background:#0f172a;'
            f'border-radius:5px;border-top:3px solid {d_color};min-width:52px;">'
            f'<div style="font-size:10px;color:#94a3b8;">{d_label}</div>'
            f'<div style="font-size:13px;font-weight:700;color:{d_color};">'
            f'{"—" if dn == 0 else f"{dw}-{dl}"}</div></div>'
        )
    day_html += '</div>'

    st.markdown(
        f'<div style="background:#1e293b;border:1px solid #1e40af;border-radius:10px;'
        f'padding:14px 18px;margin-bottom:16px;">'
        f'<div style="color:#38bdf8;font-weight:700;font-size:15px;margin-bottom:8px;">'
        f'{label} &nbsp;·&nbsp; {wk_w}-{wk_l} ({wk_wr}) &nbsp;·&nbsp; '
        f'<span style="color:#94a3b8;font-size:13px;">{wk_n} decided</span></div>'
        + day_html
        + build_table(wk_df)
        + '</div>',
        unsafe_allow_html=True
    )
