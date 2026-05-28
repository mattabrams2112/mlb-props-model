"""
Weekly Tracker — rating band × projection breakdown for each week.
One card per week, newest first.
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
st.caption('Rating band × projection breakdown per week — newest first.')

if st.button('🔄 Refresh', use_container_width=False):
    st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

df = load_all()
if df.empty:
    st.info('No play data yet. Open Game View to start logging plays.')
    st.stop()

df['rating']    = pd.to_numeric(df['rating'],    errors='coerce')
df['projected'] = pd.to_numeric(df['projected'], errors='coerce')
df['actual']    = pd.to_numeric(df['actual'],    errors='coerce')
df['date_str']  = df['date'].astype(str).str[:10]

# Clear today's results (may be mid-game)
today_str = datetime.now().strftime('%Y-%m-%d')
df = df.copy()
df.loc[df['date_str'] >= today_str, ['actual', 'result']] = ''

decided_all = df[df['result'].isin(['W', 'L'])]

# ── Week helpers ──────────────────────────────────────────────────────────────

def week_start(date_str: str) -> str:
    """Return Monday of the week containing date_str (YYYY-MM-DD)."""
    d = datetime.strptime(date_str, '%Y-%m-%d')
    return (d - timedelta(days=d.weekday())).strftime('%Y-%m-%d')

def week_label(monday_str: str) -> str:
    mon = datetime.strptime(monday_str, '%Y-%m-%d')
    sun = mon + timedelta(days=6)
    return f"{mon.strftime('%b %-d')} – {sun.strftime('%b %-d, %Y')}"

def win_rate(sub):
    d = sub[sub['result'].isin(['W', 'L'])]
    if len(d) == 0:
        return None, 0, 0, 0
    w = int((d['result'] == 'W').sum())
    l = int((d['result'] == 'L').sum())
    return round(w / len(d) * 100, 1), len(d), w, l

def roi(sub):
    d = sub[sub['result'].isin(['W', 'L'])]
    if len(d) == 0:
        return None
    w = (d['result'] == 'W').sum()
    l = (d['result'] == 'L').sum()
    return round((w * 100 - l * 110) / (len(d) * 110) * 100, 1)

def color_wr(wr):
    if wr is None: return '#475569'
    if wr >= 60:   return '#22c55e'
    if wr >= 52:   return '#eab308'
    return '#ef4444'

RATING_BUCKETS = [
    (60, 65, '60-64'), (65, 70, '65-69'), (70, 75, '70-74'),
    (75, 80, '75-79'), (80, 85, '80-84'), (85, 90, '85-89'),
    (90, 95, '90-94'), (95, 101, '95+'),
]
PROJ_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

# ── Build week list ───────────────────────────────────────────────────────────

df['week'] = df['date_str'].apply(week_start)
weeks = sorted(df['week'].dropna().unique(), reverse=True)

if not weeks:
    st.info('No plays with valid dates found.')
    st.stop()

# ── Overall season summary ────────────────────────────────────────────────────

st.markdown('### Season Totals')
s_wr, s_n, s_w, s_l = win_rate(df)
s_roi = roi(df)
pct_str = f'{s_wr}%' if s_wr else '—'
roi_str = f'{s_roi}%' if s_roi is not None else '—'

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Total Plays', s_n)
c2.metric('Record', f'{s_w}-{s_l}')
c3.metric('Win Rate', pct_str)
c4.metric('ROI (-110)', roi_str)
c5.metric('Weeks Tracked', len(weeks))

st.markdown('---')

# ── One card per week ─────────────────────────────────────────────────────────

for week_mon in weeks:
    week_df  = df[df['week'] == week_mon]
    wk_wr, wk_n, wk_w, wk_l = win_rate(week_df)
    wk_roi   = roi(week_df)
    wk_label = week_label(week_mon)
    wk_color = color_wr(wk_wr)
    wr_str   = f'{wk_wr}%' if wk_wr else '—'
    roi_disp = f'{wk_roi}%' if wk_roi is not None else '—'
    rec_str  = f'{wk_w}-{wk_l}'

    # ── Build rating band table for this week ─────────────────────────────────
    band_rows = []
    has_data  = False
    for lo, hi, label in RATING_BUCKETS:
        band = week_df[(week_df['rating'] >= lo) & (week_df['rating'] < hi)]
        total_decided = len(band[band['result'].isin(['W', 'L'])])
        row = {'Rating Band': label, 'Total Plays': total_decided}
        for pt in PROJ_THRESHOLDS:
            sub = band[band['projected'] >= pt]
            _, n, w, l = win_rate(sub)
            if n > 0:
                wr_val = round(w / n * 100, 1)
                row[f'Proj ≥{pt}'] = f"{wr_val}% ({w}-{l})"
                has_data = True
            else:
                row[f'Proj ≥{pt}'] = '—'
        band_rows.append(row)

    band_df = pd.DataFrame(band_rows)

    # ── Render the week card ──────────────────────────────────────────────────
    with st.expander(
        f"**{wk_label}** — {rec_str} ({wr_str}) · ROI {roi_disp}",
        expanded=(week_mon == weeks[0])   # open the most recent week
    ):
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Record',    rec_str)
        m2.metric('Win Rate',  wr_str)
        m3.metric('ROI',       roi_disp)
        m4.metric('Plays',     wk_n)

        if has_data:
            st.markdown('##### Rating Band × Projection')

            # Color cells in the table via HTML
            col_headers = ['Rating Band', 'Total Plays'] + [f'Proj ≥{pt}' for pt in PROJ_THRESHOLDS]

            table_html = (
                '<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:12px;">'
                '<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:11px;">'
            )
            for h in col_headers:
                table_html += f'<th style="padding:7px 10px;text-align:{"left" if h=="Rating Band" else "center"};">{h}</th>'
            table_html += '</tr></thead><tbody>'

            for _, row in band_df.iterrows():
                table_html += '<tr style="border-bottom:1px solid #1e293b;">'
                for col in col_headers:
                    val  = str(row[col])
                    align = 'left' if col == 'Rating Band' else 'center'
                    # Color win-rate cells
                    cell_color = '#e0f2fe'
                    if col not in ('Rating Band', 'Total Plays') and val != '—':
                        try:
                            pct = float(val.split('%')[0])
                            cell_color = '#22c55e' if pct >= 60 else '#eab308' if pct >= 52 else '#ef4444'
                        except Exception:
                            pass
                    table_html += (
                        f'<td style="padding:7px 10px;text-align:{align};'
                        f'color:{cell_color};font-weight:{"700" if col not in ("Rating Band","Total Plays") and val != "—" else "400"};">'
                        f'{val}</td>'
                    )
                table_html += '</tr>'

            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)

            # Day-by-day breakdown inside the week
            st.markdown('')
            st.markdown('##### Daily Breakdown')
            days = sorted(week_df['date_str'].unique())
            day_cols = st.columns(len(days)) if len(days) <= 7 else st.columns(7)
            for i, day in enumerate(days):
                day_sub = week_df[week_df['date_str'] == day]
                _, dn, dw, dl = win_rate(day_sub)
                col_idx = i % 7
                d_label = datetime.strptime(day, '%Y-%m-%d').strftime('%a %-d')
                d_color = color_wr(round(dw/dn*100,1) if dn > 0 else None)
                day_cols[col_idx].markdown(
                    f'<div style="text-align:center;padding:6px;background:#1e293b;'
                    f'border-radius:6px;border-top:3px solid {d_color};">'
                    f'<div style="font-size:11px;color:#94a3b8;">{d_label}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:{d_color};">'
                    f'{"—" if dn == 0 else f"{dw}-{dl}"}</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info('No decided plays this week yet.')

    st.markdown('')
