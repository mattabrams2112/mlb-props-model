"""
Tracker — logs qualifying predictions and tracks W/L record.
Criteria: Rating >= 85
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
from full_tracker import log_play as _log_play_ft, load_all as _load_all_ft
from odds_api import get_todays_event_ids, get_hrr_lines, ODDS_API_KEY
from ratings_cache import _load as load_ratings_cache
from shared_styles import inject_styles

st.set_page_config(page_title="Tracker | MLB Props", page_icon="📊", layout="wide")
inject_styles()

MLB_API = 'https://statsapi.mlb.com/api/v1'


def auto_fill_lines(df: pd.DataFrame) -> tuple:
    """Fetch missing lines from The Odds API for today's pending plays."""
    if not ODDS_API_KEY:
        return df, 0

    today   = datetime.now().strftime('%Y-%m-%d')
    pending = df[(df['date'] == today) &
                 (df['line'].isna() | (df['line'].astype(str).str.strip() == ''))]

    if pending.empty:
        return df, 0

    # Get today's event map once
    event_map = get_todays_event_ids()
    if not event_map:
        return df, 0

    from lineup_fetcher import TEAM_ABBR
    NICKNAMES = {v: k.split()[-1] for k, v in TEAM_ABBR.items()}

    filled = 0
    for i, row in pending.iterrows():
        team = str(row.get('team', ''))
        nickname = NICKNAMES.get(team, '')
        event_id = event_map.get(team) or event_map.get(nickname) or ''
        if not event_id:
            continue
        lines = get_hrr_lines(event_id)
        if not lines:
            continue
        # Match player name
        from odds_api import match_player
        matched = match_player(row['player'], list(lines.keys()))
        if matched:
            entry = lines[matched]
            df.at[i, 'line']      = str(entry['line'])
            df.at[i, 'over_odds'] = str(entry['over_odds'])
            filled += 1

    return df, filled


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
    """Fetch actuals for past days only — never today. Uses boxscores (fast)."""
    from full_tracker import _get_boxscore_stats_for_date
    updated = 0
    df      = df.copy()
    today   = datetime.now().strftime('%Y-%m-%d')

    # Only fetch actuals for completed past days — never today's games
    pending = df[
        (df['actual'].astype(str).str.strip().isin(['', 'nan'])) &
        (df['date'].astype(str).str[:10] < today)
    ]
    if pending.empty:
        return df, 0

    for game_date in pending['date'].astype(str).str[:10].unique():
        player_stats = _get_boxscore_stats_for_date(game_date)
        if not player_stats:
            continue
        date_rows = df[df['date'].astype(str).str[:10] == game_date]
        for i in date_rows.index:
            row = df.loc[i]
            if str(row.get('actual', '')).strip() not in ('', 'nan'):
                continue
            player_lower = str(row.get('player', '')).lower().strip()
            hrr = player_stats.get(player_lower)
            if hrr is None:
                last = player_lower.split()[-1] if player_lower else ''
                for k, v in player_stats.items():
                    if last and last in k:
                        hrr = v
                        break
            if hrr is not None:
                df.at[i, 'actual'] = str(hrr)
                # Store 1.5 as default line if none set
                line_val = str(row.get('line', '')).strip()
                if not line_val or line_val in ('nan', ''):
                    df.at[i, 'line'] = '1.5'
                updated += 1

    if updated:
        df = recalc_results(df)
    return df, updated


# Auto-import qualifying players from today's lineup
if 'lineup_rows' in st.session_state:
    qualified = [r for r in st.session_state['lineup_rows'] if r['Rating'] >= 85]
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

# Auto-sync qualifying plays from ratings cache on page load (once per session)
def sync_from_ratings_cache():
    """Pull qualifying plays from ratings cache for all recent dates."""
    ratings = load_ratings_cache()
    if ratings.empty:
        return 0

    today = datetime.now().strftime('%Y-%m-%d')
    _r = pd.to_numeric(ratings['rating'], errors='coerce')
    qualifying = ratings[
        (ratings['date'].astype(str).str[:10] <= today) &
        (_r >= 85) &
        (ratings['player_name'].astype(str).str.strip() != '')
    ]
    if qualifying.empty:
        return 0

    # Load full play log once for batch comparison
    fpl = _load_all_ft()
    fpl_keys = set()
    if not fpl.empty:
        fpl_keys = set(
            fpl['date'].astype(str).str[:10] + '|' + fpl['player'].astype(str)
        )

    total_added = 0
    new_fpl_rows = []
    for game_date in qualifying['date'].unique():
        rows = []
        for _, r in qualifying[qualifying['date'] == game_date].iterrows():
            rows.append({
                'player':     r['player_name'],
                'team':       r.get('team', ''),
                'rating':     int(r['rating']),
                'grade':      r.get('grade', ''),
                'projected':  float(r['projected']),
                'vs_pitcher': r.get('vs_pitcher', ''),
            })
        if rows:
            total_added += add_predictions(rows, game_date=game_date)
            # Collect rows missing from full play log for batch write
            date_str = str(game_date)[:10]
            for r in rows:
                key = f"{date_str}|{r['player']}"
                if key not in fpl_keys:
                    new_fpl_rows.append({
                        'date': date_str, 'player': r['player'],
                        'team': r['team'], 'rating': r['rating'],
                        'grade': r['grade'], 'projected': r['projected'],
                        'line': '', 'over_odds': '', 'actual': '',
                        'result': '', 'vs_pitcher': r['vs_pitcher'], 'is_home': 1,
                    })
                    fpl_keys.add(key)

    # Batch write new rows to full play log in one operation
    if new_fpl_rows:
        from full_tracker import save_all as _save_all_ft
        new_df = pd.DataFrame(new_fpl_rows)
        combined = pd.concat([fpl, new_df], ignore_index=True) if not fpl.empty else new_df
        _save_all_ft(combined)

    return total_added

if 'tracker_cache_synced' not in st.session_state:
    synced = sync_from_ratings_cache()
    st.session_state['tracker_cache_synced'] = True
    if synced > 0:
        df = load()

# Clear any actuals for today's games (may have been fetched mid-game)
_today = datetime.now().strftime('%Y-%m-%d')
_today_rows = df['date'].astype(str).str[:10] >= _today
if _today_rows.any():
    df = df.copy()
    df.loc[_today_rows, 'actual'] = ''
    df.loc[_today_rows, 'result'] = ''
    save(df)

# Auto-fill missing lines from Odds API on page load
if 'tracker_lines_filled' not in st.session_state:
    df, n_filled = auto_fill_lines(df)
    if n_filled > 0:
        save(df)
    st.session_state['tracker_lines_filled'] = True

_hdr, _btn = st.columns([5, 1])
with _hdr:
    st.markdown('## 📊 Prediction Tracker')
    st.caption('Criteria: Rating ≥ 75 · Lines entered manually · Actuals fetched automatically')
with _btn:
    if st.button('🔄 Refresh', use_container_width=True):
        st.rerun()

# Filter to current criteria only
df['_r'] = pd.to_numeric(df['rating'], errors='coerce')
df = df[df['_r'] >= 85].copy()
df.drop(columns=['_r'], inplace=True)

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

# ── Pending breakdown ──────────────────────────────────────────────────────────
_pending_df = df[df['result'] == ''].copy()
_past_pending = _pending_df[_pending_df['date'].astype(str).str[:10] < _today]

if not _past_pending.empty:
    st.warning(f'⚠️ {len(_past_pending)} unresolved play(s) from past dates — hit **Auto-fetch Actuals** or correct manually.')
    with st.expander(f'📋 View {len(_past_pending)} past pending plays', expanded=True):
        _show = _past_pending[['date','player','team','rating','projected','line','actual']].copy()
        _show['date'] = _show['date'].astype(str).str[:10]
        st.dataframe(_show.sort_values('date', ascending=False), hide_index=True, use_container_width=True)
        st.caption('Mark postponed games to remove them from pending without counting as W or L.')
        for _pi, _pr in _past_pending.iterrows():
            _pcol1, _pcol2 = st.columns([4, 1])
            with _pcol1:
                st.markdown(f'`{str(_pr["date"])[:10]}` **{_pr["player"]}** ({_pr["team"]})')
            with _pcol2:
                if st.button('🌧️ PPD', key=f'ppd_{_pi}', use_container_width=True):
                    from tracker import load as _tload, save as _tsave
                    _tdf = _tload()
                    _tmask = ((_tdf['player'] == _pr['player']) &
                              (_tdf['date'].astype(str).str[:10] == str(_pr['date'])[:10]))
                    _tdf.loc[_tmask, 'result'] = 'PPD'
                    _tdf.loc[_tmask, 'actual'] = 'PPD'
                    _tsave(_tdf)
                    from full_tracker import load_all as _fload, save_all as _fsave
                    _fdf = _fload()
                    _fmask = ((_fdf['player'] == _pr['player']) &
                               (_fdf['date'].astype(str).str[:10] == str(_pr['date'])[:10]))
                    _fdf.loc[_fmask, 'result'] = 'PPD'
                    _fdf.loc[_fmask, 'actual'] = 'PPD'
                    _fsave(_fdf)
                    st.rerun()

if not df.empty and len(_pending_df) > 0:
    _today_pending = _pending_df[_pending_df['date'].astype(str).str[:10] >= _today]
    if not _today_pending.empty:
        st.info(f'📅 {len(_today_pending)} pending play(s) from today — results will be available after games finish.')

st.markdown('---')

if df.empty:
    st.info('No predictions tracked yet. Open the **🎯 Game View** and let the lineups fully load — qualifying plays are added automatically.')
    st.stop()

# ── Manual add play ───────────────────────────────────────────────────────────

with st.expander('➕ Manually Add a Play', expanded=False):
    st.caption('Adds to both the Tracker and Daily Results simultaneously.')
    from full_tracker import log_play as _log_play
    ma_col1, ma_col2 = st.columns(2)
    with ma_col1:
        ma_player  = st.text_input('Player Name', key='ma_player')
        ma_team    = st.text_input('Team (abbr)', key='ma_team')
        ma_rating  = st.number_input('Rating', min_value=0, max_value=100, value=75, key='ma_rating')
        ma_proj    = st.number_input('Projected HRR', min_value=0.0, max_value=10.0, value=2.0, step=0.1, key='ma_proj')
    with ma_col2:
        ma_date    = st.date_input('Date', key='ma_date')
        ma_pitcher = st.text_input('vs Pitcher', key='ma_pitcher')
        ma_line    = st.number_input('Line', min_value=0.0, max_value=10.0, value=1.5, step=0.5, key='ma_line')
        ma_actual  = st.number_input('Actual HRR (leave 0 if pending)', min_value=0, max_value=20, value=0, key='ma_actual')

    if st.button('✅ Add Play', type='primary', key='ma_submit'):
        if ma_player and ma_team:
            ma_date_str = ma_date.strftime('%Y-%m-%d')
            ma_grade = ('A+' if ma_rating >= 90 else 'A' if ma_rating >= 85 else 'A-' if ma_rating >= 80
                        else 'B+' if ma_rating >= 75 else 'B' if ma_rating >= 70 else 'B-')
            ma_result = ''
            if ma_actual > 0 and ma_date_str < _today:
                ma_result = 'W' if ma_actual > ma_line else 'L'

            # Write to full_play_log (Daily Results)
            try:
                _log_play(
                    player=ma_player, team=ma_team, rating=ma_rating,
                    grade=ma_grade, projected=ma_proj, line=ma_line,
                    vs_pitcher=ma_pitcher, game_date=ma_date_str,
                )
                if ma_actual > 0:
                    from full_tracker import load_all as _load_all, save_all as _save_all
                    _fdf = _load_all()
                    _mask = (_fdf['player'] == ma_player) & (_fdf['date'].astype(str).str[:10] == ma_date_str)
                    if _mask.any():
                        _fdf.loc[_mask, 'actual'] = str(ma_actual)
                        _fdf.loc[_mask, 'result'] = ma_result
                        _save_all(_fdf)
            except Exception as e:
                st.error(f'Daily Results error: {e}')

            # Write to tracker
            try:
                add_predictions([{
                    'player': ma_player, 'team': ma_team,
                    'rating': ma_rating, 'grade': ma_grade,
                    'projected': ma_proj, 'line': ma_line,
                    'vs_pitcher': ma_pitcher,
                }], game_date=ma_date_str)
                if ma_actual > 0:
                    tdf = load()
                    _tmask = (tdf['player'] == ma_player) & (tdf['date'].astype(str).str[:10] == ma_date_str)
                    if _tmask.any():
                        tdf.loc[_tmask, 'actual'] = str(ma_actual)
                        tdf.loc[_tmask, 'result'] = ma_result
                        save(tdf)
            except Exception as e:
                st.error(f'Tracker error: {e}')

            st.success(f'✅ Added {ma_player} ({ma_date_str}) to both Tracker and Daily Results!')
            st.rerun()
        else:
            st.warning('Player name and team are required.')

st.markdown('---')

# ── Auto-fetch actuals ─────────────────────────────────────────────────────────

col_sync, col_fetch = st.columns(2)
with col_sync:
    if st.button('🔁 Sync Today\'s Plays', use_container_width=True,
                 help='Pull any missing qualifying plays from the ratings cache'):
        st.session_state.pop('tracker_synced', None)
        n = sync_from_ratings_cache()
        df = load()
        st.success(f'Synced {n} new play(s)!' if n else 'All plays already tracked.')
        st.rerun()

with col_fetch:
    if st.button('🔄 Auto-fetch Actuals from MLB API', type='primary', use_container_width=True):
        today_str = datetime.now().strftime('%Y-%m-%d')
        pending = df[df['actual'].isna() | (df['actual'].astype(str).str.strip().isin(['', 'nan']))]
        st.caption(f'Debug: {len(df)} total plays, {len(pending)} missing actuals, today={today_str}')
        if not pending.empty:
            st.caption(f'Sample dates: {pending["date"].tolist()[:5]}')
        with st.spinner('Fetching actual H+R+RBI for completed games...'):
            df, updated = auto_fill_actuals(df)
        if updated:
            save(df)
            st.success(f'Updated {updated} player(s) with actual results!')
            st.rerun()
        else:
            st.info('No new actuals to fetch — either games are pending or already filled.')

# ── Manual correction ─────────────────────────────────────────────────────────

with st.expander('✏️ Manually Correct an Actual', expanded=False):
    st.caption('Use this to fix wrong actuals (e.g. mid-game stats fetched incorrectly).')
    dates   = sorted(df['date'].astype(str).str[:10].unique(), reverse=True)
    sel_date = st.selectbox('Date', dates, key='manual_date')
    day_df   = df[df['date'].astype(str).str[:10] == sel_date]
    players  = day_df['player'].tolist()
    sel_player = st.selectbox('Player', players, key='manual_player')
    new_actual = st.number_input('Correct Actual H+R+RBI', min_value=0, max_value=20, step=1, key='manual_actual')
    if st.button('✅ Apply Correction', type='primary'):
        idx = df[(df['player'] == sel_player) & (df['date'].astype(str).str[:10] == sel_date)].index
        if len(idx) > 0:
            df = df.copy()
            df.loc[idx[0], 'actual'] = str(new_actual)
            line_val = str(df.loc[idx[0], 'line']).strip()
            line = float(line_val) if line_val and line_val not in ('nan', '') else 1.5
            df.loc[idx[0], 'result'] = 'W' if new_actual > line else 'L'
            save(df)
            st.success(f'Updated {sel_player} ({sel_date}): actual={new_actual}, result={"W" if new_actual > line else "L"}')
            st.rerun()

# ── Remove a play ─────────────────────────────────────────────────────────────

with st.expander('🗑️ Remove a Play', expanded=False):
    st.caption('Removes a play from both the tracker and the full play log (Daily Results / Analytics).')
    from full_tracker import load_all as _fload, save_all as _fsave
    _all_dates  = sorted(df['date'].astype(str).str[:10].unique(), reverse=True)
    _rem_date   = st.selectbox('Date', _all_dates, key='rem_date')
    _day_plays  = df[df['date'].astype(str).str[:10] == _rem_date]['player'].tolist()
    if _day_plays:
        _rem_player = st.selectbox('Player', _day_plays, key='rem_player')
        st.warning(f'This will permanently delete **{_rem_player}** ({_rem_date}) from all tables.')
        if st.button('🗑️ Remove Play', type='primary', key='rem_btn'):
            # Remove from tracker
            _new_df = df[~((df['player'] == _rem_player) & (df['date'].astype(str).str[:10] == _rem_date))].copy()
            save(_new_df)
            # Remove from full play log
            _fdf = _fload()
            _fdf = _fdf[~((_fdf['player'] == _rem_player) & (_fdf['date'].astype(str).str[:10] == _rem_date))].reset_index(drop=True)
            _fsave(_fdf)
            st.success(f'Removed {_rem_player} ({_rem_date}) from all tables.')
            st.rerun()
    else:
        st.info('No plays on this date.')

# ── Editable table ─────────────────────────────────────────────────────────────

st.markdown('### Results')
st.caption('Lines must be entered manually. Click **Auto-fetch Actuals** to pull real H+R+RBI after games finish. W/L calculates automatically.')

df['_sort'] = df['result'].apply(lambda x: 0 if x == '' else 1)
df = df.sort_values(['_sort', 'date', 'rating'], ascending=[True, False, False]).drop(columns=['_sort'])

def _bet_size(rating):
    return '$8 (1u)'

df['bet'] = df['rating'].apply(_bet_size)

edited = st.data_editor(
    df,
    column_config={
        'date':       st.column_config.TextColumn('Date',         disabled=True, width='small'),
        'player':     st.column_config.TextColumn('Player',       disabled=True, width='medium'),
        'team':       st.column_config.TextColumn('Team',         disabled=True, width='small'),
        'rating':     st.column_config.NumberColumn('Rating',     disabled=True, width='small'),
        'grade':      st.column_config.TextColumn('Grade',        disabled=True, width='small'),
        'projected':  st.column_config.NumberColumn('Proj HRR',   disabled=True, width='small', format='%.2f'),
        'bet':        st.column_config.TextColumn('💰 Bet',        disabled=True, width='small'),
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
            day_display['bet'] = day_display['rating'].apply(_bet_size)
            day_display['result'] = day_display['result'].map({'W': '✅ W', 'L': '❌ L'})
            st.dataframe(
                day_display.rename(columns={
                    'player': 'Player', 'team': 'Team', 'rating': 'Rating',
                    'projected': 'Proj', 'bet': '💰 Bet', 'line': 'Line', 'actual': 'Actual',
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
