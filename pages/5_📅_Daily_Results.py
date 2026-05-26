"""
Daily Results — day-by-day performance filtered by current tracking criteria.
Criteria: Rating 70-74 + Proj >= 3.0  OR  Rating 75-89 + Proj >= 1.5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime
from full_tracker import load_all, update_actuals, save_all

st.set_page_config(page_title="Daily Results | MLB Props", page_icon="📅", layout="wide")

st.markdown("""
<style>
  h1,h2,h3{color:#38bdf8!important;}
  .stMarkdown p,label,.stCaption{color:#7dd3fc!important;}
  .stMetric label{color:#38bdf8!important;}
  .stMetric [data-testid="metric-container"]>div{color:#e0f2fe!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('## 📅 Daily Results')
st.caption('Day-by-day performance for plays matching current criteria: Rating 70–74 + Proj ≥ 3.0 OR Rating 75–89 + Proj ≥ 1.5')

col_refresh, col_fetch = st.columns([1, 1])
with col_refresh:
    if st.button('🔄 Refresh', use_container_width=True):
        st.rerun()
with col_fetch:
    if st.button('⬇️ Fetch Latest Results', type='primary', use_container_width=True):
        with st.spinner('Fetching results from MLB API...'):
            n = update_actuals()
        st.success(f'Updated {n} plays!')
        st.rerun()

df = load_all()

if df.empty:
    st.info('No play data yet. Open the Game View page to start logging plays automatically.')
    st.stop()

# Numeric types
df['rating']    = pd.to_numeric(df['rating'],    errors='coerce')
df['projected'] = pd.to_numeric(df['projected'], errors='coerce')
df['actual']    = pd.to_numeric(df['actual'],    errors='coerce')
df['date_str']  = df['date'].astype(str).str[:10]

# Apply current criteria
today_str = datetime.now().strftime('%Y-%m-%d')
criteria = df[
    (
        ((df['rating'] >= 70) & (df['rating'] <= 74) & (df['projected'] >= 3.0)) |
        ((df['rating'] >= 75) & (df['rating'] <= 89) & (df['projected'] >= 1.5))
    )
]

decided = criteria[criteria['result'].isin(['W', 'L'])]
pending = criteria[criteria['result'] == '']

# ── Overall summary ───────────────────────────────────────────────────────────

total_w = (decided['result'] == 'W').sum()
total_l = (decided['result'] == 'L').sum()
total_n = len(decided)
total_wr = round(total_w / total_n * 100, 1) if total_n > 0 else None
total_roi = round((total_w * 100 - total_l * 110) / (total_n * 110) * 100, 1) if total_n > 0 else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Qualifying Plays', len(criteria))
c2.metric('Decided', total_n)
c3.metric('Record', f'{total_w}-{total_l}' if total_n > 0 else '—')
c4.metric('Win Rate', f'{total_wr}%' if total_wr else '—')
c5.metric('ROI', f'{total_roi}%' if total_roi is not None else '—')

st.markdown('---')

# ── Day-by-day table ──────────────────────────────────────────────────────────

st.markdown('### Day-by-Day Breakdown')

if decided.empty:
    st.info('No decided plays yet for current criteria.')
else:
    daily_rows = []
    for date, grp in decided.groupby('date_str', sort=False):
        w = int((grp['result'] == 'W').sum())
        l = int((grp['result'] == 'L').sum())
        n = w + l
        wr = round(w / n * 100, 1) if n > 0 else 0
        profit = w * 100 - l * 110
        daily_rows.append({
            'Date':     date,
            'Record':   f'{w}-{l}',
            'Win Rate': f'{wr}%',
            'Profit':   f'+{profit}u' if profit >= 0 else f'{profit}u',
            'Plays':    n,
        })

    daily_df = pd.DataFrame(daily_rows).sort_values('Date', ascending=False)

    # Color-coded HTML table
    def row_color(wr_str):
        wr = float(wr_str.replace('%', ''))
        if wr >= 60:
            return '#14532d'
        if wr >= 52.4:
            return '#1c3a1a'
        if wr > 0:
            return '#450a0a'
        return '#1e293b'

    def wr_color(wr_str):
        wr = float(wr_str.replace('%', ''))
        if wr >= 60:
            return '#22c55e'
        if wr >= 52.4:
            return '#eab308'
        return '#ef4444'

    html = '''<table style="width:100%;border-collapse:collapse;font-family:monospace;">
<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:13px;">
<th style="padding:10px 12px;text-align:left;">Date</th>
<th style="padding:10px 12px;text-align:center;">Record</th>
<th style="padding:10px 12px;text-align:center;">Win Rate</th>
<th style="padding:10px 12px;text-align:center;">Profit (units)</th>
<th style="padding:10px 12px;text-align:center;">Plays</th>
</tr></thead><tbody>'''

    running_profit = 0
    for _, row in daily_df.iterrows():
        bg = row_color(row['Win Rate'])
        wrc = wr_color(row['Win Rate'])
        profit_val = float(row['Profit'].replace('u','').replace('+',''))
        running_profit += profit_val
        rp_color = '#22c55e' if running_profit >= 0 else '#ef4444'
        html += f'''<tr style="background:{bg};border-bottom:1px solid #334155;">
<td style="padding:9px 12px;color:#e0f2fe;">{row['Date']}</td>
<td style="padding:9px 12px;text-align:center;color:#e0f2fe;font-weight:700;">{row['Record']}</td>
<td style="padding:9px 12px;text-align:center;color:{wrc};font-weight:800;">{row['Win Rate']}</td>
<td style="padding:9px 12px;text-align:center;color:{"#22c55e" if profit_val >= 0 else "#ef4444"};font-weight:700;">{row['Profit']}</td>
<td style="padding:9px 12px;text-align:center;color:#94a3b8;">{row['Plays']}</td>
</tr>'''

    html += f'''<tr style="background:#0f172a;border-top:2px solid #38bdf8;">
<td style="padding:10px 12px;color:#38bdf8;font-weight:800;">TOTAL</td>
<td style="padding:10px 12px;text-align:center;color:#e0f2fe;font-weight:800;">{total_w}-{total_l}</td>
<td style="padding:10px 12px;text-align:center;color:{"#22c55e" if (total_wr or 0) >= 52.4 else "#ef4444"};font-weight:800;">{total_wr}%</td>
<td style="padding:10px 12px;text-align:center;color:{"#22c55e" if running_profit >= 0 else "#ef4444"};font-weight:800;">{"+" if running_profit >= 0 else ""}{running_profit:.0f}u</td>
<td style="padding:10px 12px;text-align:center;color:#94a3b8;">{total_n}</td>
</tr></tbody></table>'''

    st.markdown(html, unsafe_allow_html=True)

st.markdown('---')

# ── Pending plays ─────────────────────────────────────────────────────────────

if not pending.empty:
    st.markdown(f'### Pending Plays ({len(pending)})')
    st.caption("Today's qualifying plays — results not in yet.")

    pend_rows = []
    for _, row in pending.sort_values('date_str', ascending=False).iterrows():
        pend_rows.append({
            'Date':      row['date_str'],
            'Player':    row.get('player', ''),
            'Team':      row.get('team', ''),
            'Rating':    int(row['rating']) if pd.notna(row['rating']) else '—',
            'Projected': row['projected'] if pd.notna(row['projected']) else '—',
            'Line':      row.get('line', '—'),
        })

    st.dataframe(pd.DataFrame(pend_rows), hide_index=True, use_container_width=True)

st.markdown('---')

# ── Play log for current criteria ─────────────────────────────────────────────

with st.expander('📋 Full Play Log (current criteria)', expanded=False):
    show_cols = ['date_str', 'player', 'team', 'rating', 'projected', 'line', 'actual', 'result']
    show_cols = [c for c in show_cols if c in criteria.columns]
    display = criteria[show_cols].rename(columns={'date_str': 'date'}).sort_values('date', ascending=False)
    st.dataframe(display, hide_index=True, use_container_width=True)
