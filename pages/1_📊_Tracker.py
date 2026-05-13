"""
Tracker — logs all 60+ rated predictions and tracks W/L record.
WIN  = actual H+R+RBI > sportsbook line you entered.
LOSS = actual H+R+RBI ≤ sportsbook line you entered.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime
from tracker import load, save, recalc_results, add_predictions

st.set_page_config(page_title="Tracker | MLB Props", page_icon="📊", layout="wide")

# Auto-import any 60+ rated players from today's lineup
if 'lineup_rows' in st.session_state:
    qualified = [r for r in st.session_state['lineup_rows'] if r['Rating'] >= 60]
    if qualified:
        add_predictions([{
            'player':     r['Player'],
            'team':       r['_team'],
            'rating':     r['Rating'],
            'grade':      r['Grade'],
            'projected':  r['Projected'],
            'vs_pitcher': r['vs Pitcher'],
        } for r in qualified])

df = load()

st.markdown('## 📊 Prediction Tracker')
st.caption('All predictions rated 60+ are auto-added here. Enter the sportsbook line and actual H+R+RBI after each game.')

# ── Record summary ────────────────────────────────────────────────────────────

decided = df[df['result'].isin(['W', 'L'])]
wins    = int((decided['result'] == 'W').sum())
losses  = int((decided['result'] == 'L').sum())
total   = wins + losses
pct     = f"{wins / total:.0%}" if total > 0 else '—'

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Record',  f"{wins} - {losses}")
c2.metric('Win %',   pct)
c3.metric('Decided', total)
c4.metric('Pending', int((df['result'] == '').sum()))
c5.metric('Tracked', len(df))

st.markdown('---')

if df.empty:
    st.info('No predictions tracked yet. Go to the home page — players rated 60+ are added here automatically when lineups load.')
    st.stop()

# ── Editable table ────────────────────────────────────────────────────────────

st.markdown('### Enter Lines & Actuals')
st.caption('Edit the **Line** and **Actual** columns. W/L calculates automatically when you click Save.')

df['_sort'] = df['result'].apply(lambda x: 0 if x == '' else 1)
df = df.sort_values(['_sort', 'date', 'rating'], ascending=[True, False, False]).drop(columns=['_sort'])

edited = st.data_editor(
    df,
    column_config={
        'date':       st.column_config.TextColumn('Date',         disabled=True, width='small'),
        'player':     st.column_config.TextColumn('Player',       disabled=True, width='medium'),
        'team':       st.column_config.TextColumn('Team',         disabled=True, width='small'),
        'rating':     st.column_config.NumberColumn('Rating',     disabled=True, width='small'),
        'grade':      st.column_config.TextColumn('Grade',        disabled=True, width='small'),
        'projected':  st.column_config.NumberColumn('Proj HRR',   disabled=True, width='small', format='%.2f'),
        'line':       st.column_config.NumberColumn('📥 Line',    width='small', help='Sportsbook line'),
        'actual':     st.column_config.NumberColumn('✏️ Actual', width='small', help='Actual H+R+RBI after game'),
        'result':     st.column_config.TextColumn('Result',       disabled=True, width='small'),
        'vs_pitcher': st.column_config.TextColumn('vs Pitcher',   disabled=True, width='medium'),
    },
    hide_index=True,
    use_container_width=True,
    num_rows='fixed',
)

if st.button('💾 Save Changes', type='primary'):
    updated = recalc_results(edited)
    save(updated)
    st.success('Saved!')
    st.rerun()

st.markdown('---')

# ── W / L breakdown ───────────────────────────────────────────────────────────

if not decided.empty:
    st.markdown('### Results Breakdown')
    wc, lc = st.columns(2)

    with wc:
        st.markdown('**✅ Wins**')
        w_df = decided[decided['result'] == 'W'][['date','player','team','rating','projected','line','actual']].copy()
        w_df['Edge'] = w_df.apply(
            lambda r: f"+{float(r['actual']) - float(r['line']):.1f}" if r['actual'] != '' and r['line'] != '' else '', axis=1)
        st.dataframe(w_df.rename(columns={'date':'Date','player':'Player','team':'Team',
                                           'rating':'Rtg','projected':'Proj','line':'Line','actual':'Actual'}),
                     hide_index=True, use_container_width=True)

    with lc:
        st.markdown('**❌ Losses**')
        l_df = decided[decided['result'] == 'L'][['date','player','team','rating','projected','line','actual']].copy()
        l_df['Miss'] = l_df.apply(
            lambda r: f"{float(r['actual']) - float(r['line']):.1f}" if r['actual'] != '' and r['line'] != '' else '', axis=1)
        st.dataframe(l_df.rename(columns={'date':'Date','player':'Player','team':'Team',
                                           'rating':'Rtg','projected':'Proj','line':'Line','actual':'Actual'}),
                     hide_index=True, use_container_width=True)

st.markdown('---')

# ── Backup / restore ──────────────────────────────────────────────────────────

st.markdown('### Backup & Restore')
st.caption('Download your tracker data to keep a permanent backup — the file resets if the app redeploys.')

dl, ul = st.columns(2)
with dl:
    st.download_button('⬇️ Download CSV', data=df.to_csv(index=False),
                       file_name=f'mlb_tracker_{datetime.now().strftime("%Y%m%d")}.csv',
                       mime='text/csv', use_container_width=True)
with ul:
    uploaded = st.file_uploader('⬆️ Restore from CSV', type='csv')
    if uploaded:
        save(pd.read_csv(uploaded))
        st.success('Tracker restored!')
        st.rerun()
