"""
Tracker — logs all 60+ rated predictions and tracks W/L record.
WIN  = actual H+R+RBI > sportsbook line you entered.
LOSS = actual H+R+RBI ≤ sportsbook line you entered.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from tracker import load, save, recalc_results, add_predictions

st.set_page_config(page_title="Tracker | MLB Props", page_icon="📊", layout="wide")

st.markdown("""
<style>
  h1, h2, h3 { color: #38bdf8 !important; }
  .stMarkdown p, label, .stCaption { color: #7dd3fc !important; }
  .stMetric label { color: #38bdf8 !important; }
  .stMetric [data-testid="metric-container"] > div { color: #e0f2fe !important; }
</style>
""", unsafe_allow_html=True)

MLB_API = 'https://statsapi.mlb.com/api/v1'


def fetch_actual_hrr(player_name: str, game_date: str) -> float | None:
    """Fetch a player's actual H+R+RBI for a given date from the MLB API."""
    try:
        import statsapi
        players = statsapi.lookup_player(player_name)
        if not players:
            return None
        player_id = players[0]['id']

        year = game_date[:4]
        resp = requests.get(
            f'{MLB_API}/people/{player_id}/stats',
            params={'stats': 'gameLog', 'group': 'hitting', 'season': year},
            timeout=15
        )
        resp.raise_for_status()
        splits = (resp.json().get('stats') or [{}])[0].get('splits', [])

        for split in splits:
            gi   = split.get('game', {})
            gdate = gi.get('gameDate', split.get('date', ''))[:10]
            if gdate == game_date:
                stat = split.get('stat', {})
                h   = int(stat.get('hits', 0))
                r   = int(stat.get('runs', 0))
                rbi = int(stat.get('rbi', 0))
                return h + r + rbi
    except Exception:
        pass
    return None


def auto_fill_actuals(df: pd.DataFrame) -> tuple:
    """Fetch actuals for all rows that have a line but no actual yet."""
    updated = 0
    df = df.copy()
    today = datetime.now().strftime('%Y-%m-%d')

    for i, row in df.iterrows():
        if str(row.get('actual', '')).strip() not in ('', 'nan') :
            continue
        if str(row.get('line', '')).strip() in ('', 'nan'):
            continue
        game_date = str(row.get('date', ''))[:10]
        if game_date >= today:
            continue  # game hasn't happened yet

        actual = fetch_actual_hrr(row['player'], game_date)
        if actual is not None:
            df.at[i, 'actual'] = actual
            updated += 1

    if updated:
        df = recalc_results(df)
    return df, updated


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
st.caption('All predictions rated 60+ are auto-added. Lines are entered manually. Actuals are fetched automatically after games finish.')

# ── Record summary ─────────────────────────────────────────────────────────────

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

# ── Auto-fetch actuals ─────────────────────────────────────────────────────────

col_fetch, col_save = st.columns([2, 1])
with col_fetch:
    if st.button('🔄 Auto-fetch Actuals from MLB API', type='primary', use_container_width=True):
        with st.spinner('Fetching actual H+R+RBI for completed games...'):
            df, updated = auto_fill_actuals(df)
        if updated:
            save(df)
            st.success(f'Updated {updated} player(s) with actual results!')
            st.rerun()
        else:
            st.info('No new actuals to fetch — either games are pending or already filled.')

# ── Editable table ─────────────────────────────────────────────────────────────

st.markdown('### Results')
st.caption('Lines must be entered manually. Click **Auto-fetch Actuals** to pull real H+R+RBI after games finish. W/L calculates automatically.')

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
        'line':       st.column_config.NumberColumn('📥 Line',      width='small', help='Sportsbook line'),
        'over_odds':  st.column_config.NumberColumn('📊 Over Odds', width='small', disabled=True, help='Book over odds'),
        'actual':     st.column_config.NumberColumn('✏️ Actual',    width='small', help='Actual H+R+RBI (auto-filled after game)'),
        'result':     st.column_config.TextColumn('Result',       disabled=True, width='small'),
        'vs_pitcher': st.column_config.TextColumn('vs Pitcher',   disabled=True, width='medium'),
    },
    hide_index=True,
    use_container_width=True,
    num_rows='fixed',
)

if st.button('💾 Save Changes', type='secondary', use_container_width=False):
    updated = recalc_results(edited)
    save(updated)
    st.success('Saved!')
    st.rerun()

st.markdown('---')

# ── Daily breakdown ────────────────────────────────────────────────────────────

st.markdown('### Results by Day')

if not decided.empty:
    for date in sorted(decided['date'].unique(), reverse=True):
        day_df  = decided[decided['date'] == date]
        day_w   = int((day_df['result'] == 'W').sum())
        day_l   = int((day_df['result'] == 'L').sum())
        day_pct = f"{day_w / (day_w + day_l):.0%}" if (day_w + day_l) > 0 else '—'
        color   = '#22c55e' if day_w > day_l else '#ef4444' if day_l > day_w else '#eab308'

        with st.expander(f"**{date}** — {day_w}-{day_l} ({day_pct})", expanded=(date == sorted(decided['date'].unique(), reverse=True)[0])):
            day_display = day_df[['player', 'team', 'rating', 'projected', 'line', 'actual', 'result', 'vs_pitcher']].copy()
            day_display['result'] = day_display['result'].map({'W': '✅ W', 'L': '❌ L'})
            st.dataframe(
                day_display.rename(columns={
                    'player': 'Player', 'team': 'Team', 'rating': 'Rating',
                    'projected': 'Proj', 'line': 'Line', 'actual': 'Actual',
                    'result': 'Result', 'vs_pitcher': 'vs Pitcher'
                }),
                hide_index=True, use_container_width=True
            )
else:
    st.caption('No completed results yet — enter lines and click Auto-fetch Actuals after games finish.')

st.markdown('---')

# ── W/L breakdown ──────────────────────────────────────────────────────────────

if not decided.empty:
    st.markdown('### All Results')
    wc, lc = st.columns(2)
    with wc:
        st.markdown('**✅ Wins**')
        w_df = decided[decided['result'] == 'W'][['date','player','team','rating','projected','line','actual']].copy()
        w_df['Edge'] = w_df.apply(lambda r: f"+{float(r['actual'])-float(r['line']):.1f}"
                                   if str(r['actual']) not in ('','nan') and str(r['line']) not in ('','nan') else '', axis=1)
        st.dataframe(w_df.rename(columns={'date':'Date','player':'Player','team':'Team',
                                           'rating':'Rtg','projected':'Proj','line':'Line','actual':'Actual'}),
                     hide_index=True, use_container_width=True)
    with lc:
        st.markdown('**❌ Losses**')
        l_df = decided[decided['result'] == 'L'][['date','player','team','rating','projected','line','actual']].copy()
        l_df['Miss'] = l_df.apply(lambda r: f"{float(r['actual'])-float(r['line']):.1f}"
                                   if str(r['actual']) not in ('','nan') and str(r['line']) not in ('','nan') else '', axis=1)
        st.dataframe(l_df.rename(columns={'date':'Date','player':'Player','team':'Team',
                                           'rating':'Rtg','projected':'Proj','line':'Line','actual':'Actual'}),
                     hide_index=True, use_container_width=True)

st.markdown('---')

# ── Backup / restore ───────────────────────────────────────────────────────────

st.markdown('### Backup & Restore')
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
