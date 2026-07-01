"""
Daily Results — day-by-day performance filtered by current tracking criteria.
Criteria: Rating >= 85
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime
from full_tracker import load_all, update_actuals, save_all, log_play
from eastern_time import today_et, today_str_et
from shared_styles import inject_styles

st.set_page_config(page_title="Daily Results | MLB Props", page_icon="📅", layout="wide")
inject_styles()

st.markdown('## 📅 Daily Results')
st.caption('Criteria: Rating ≥ 75')

col_refresh, col_fetch = st.columns([1, 1])
with col_refresh:
    if st.button('🔄 Refresh', use_container_width=True):
        st.rerun()
with col_fetch:
    if st.button('⬇️ Fetch Latest Results', type='primary', use_container_width=True):
        with st.spinner('Fetching results from MLB API...'):
            n = update_actuals()
        # Diagnose what's still unresolved on PAST days so it's not a silent no-op
        _chk = load_all()
        _chk['_ds'] = _chk['date'].astype(str).str[:10]
        _chk['_a']  = _chk['actual'].astype(str).str.strip()
        _chk['_r']  = _chk['result'].astype(str).str.strip()
        _chk['_l']  = (_chk['line'].astype(str).str.strip()
                       if 'line' in _chk.columns else '')
        _rn   = pd.to_numeric(_chk['rating'], errors='coerce')
        _past = _chk[(_chk['_ds'] < today_str_et()) & (_rn >= 85)]
        _no_actual = int(_past['_a'].isin(['', 'nan']).sum())
        _no_line   = int(((~_past['_a'].isin(['', 'nan'])) &
                          (_past['_r'].isin(['', 'nan'])) &
                          (_past['_l'].isin(['', 'nan']))).sum())
        st.session_state['dr_fetch_msg'] = (n, _no_actual, _no_line)
        st.rerun()

# Show the outcome of the last fetch (survives the rerun above)
if 'dr_fetch_msg' in st.session_state:
    _n, _na, _nl = st.session_state.pop('dr_fetch_msg')
    st.success(f'Fetched actuals for {_n} play(s).')
    if _nl:
        st.warning(f'⚠️ {_nl} past play(s) have a final result but **no line** — '
                   f'enter a line under "Needs a Line to Grade" below to score them.')
    if _na:
        st.info(f'ℹ️ {_na} past play(s) still have no actual — the player name '
                f'may not have matched the boxscore. Use "Manually Correct" if needed.')

def _sync_tracker_to_fpl():
    """
    Keep full_play_log (Daily Results) in sync with the tracker.
      - Add tracker plays that aren't in full_play_log yet.
      - Backfill line/over_odds onto EXISTING full_play_log rows when the
        tracker has a value but full_play_log is still blank (e.g. the line
        was auto-filled or entered on the Tracker after the play was logged).
    Only writes when something actually changed.
    """
    try:
        from tracker import load as _load_tracker
        tracker_df = _load_tracker()
        if tracker_df.empty:
            return
        fpl = load_all()
        changed = False

        def _blank(v):
            return str(v).strip() in ('', 'nan', 'None')

        # Map date|player -> first full_play_log index for backfilling
        idx_by_key = {}
        if not fpl.empty:
            _keys = fpl['date'].astype(str).str[:10] + '|' + fpl['player'].astype(str)
            for _i, _k in zip(fpl.index, _keys):
                idx_by_key.setdefault(_k, _i)

        new_rows = []
        for _, row in tracker_df.iterrows():
            key = f"{str(row['date'])[:10]}|{row['player']}"
            if key in idx_by_key:
                i = idx_by_key[key]
                # Backfill line / odds from tracker when full_play_log is missing them
                for _col in ('line', 'over_odds'):
                    tv = row.get(_col, '')
                    if not _blank(tv) and (_col not in fpl.columns or _blank(fpl.at[i, _col])):
                        fpl.at[i, _col] = tv
                        changed = True
            else:
                new_rows.append({
                    'date':           str(row['date'])[:10],
                    'player':         row.get('player', ''),
                    'team':           row.get('team', ''),
                    'rating':         row.get('rating', ''),
                    'grade':          row.get('grade', ''),
                    'projected':      row.get('projected', ''),
                    'base_proj':      '',
                    'line':           row.get('line', ''),
                    'over_odds':      row.get('over_odds', ''),
                    'actual':         row.get('actual', ''),
                    'result':         row.get('result', ''),
                    'vs_pitcher':     row.get('vs_pitcher', ''),
                    'is_home':        '',
                    'pitcher_throws': '',
                })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            fpl = pd.concat([fpl, new_df], ignore_index=True) if not fpl.empty else new_df
            changed = True
        if changed:
            save_all(fpl)
    except Exception:
        pass

# Run every load (cheap; only writes when something changed) so a line entered
# on the Tracker shows up here without needing a fresh session.
_sync_tracker_to_fpl()

df = load_all()

if df.empty:
    st.info('No play data yet. Open the Game View page to start logging plays automatically.')
    st.stop()

# Numeric types
df['rating']    = pd.to_numeric(df['rating'],    errors='coerce')
df['projected'] = pd.to_numeric(df['projected'], errors='coerce')
df['actual']    = pd.to_numeric(df['actual'],    errors='coerce')
df['date_str']  = df['date'].astype(str).str[:10]

# ── Date filter ───────────────────────────────────────────────────────────────

from datetime import datetime as _dt, timedelta as _td

_today_dt           = _dt.now().date()
_current_week_start = _today_dt - _td(days=_today_dt.weekday())

_PERIODS = ['All Time', 'This Week', 'Last Week', 'Last 2 Weeks',
            'This Month', 'Last Month', 'Custom Range']

_fc1, _fc2 = st.columns([2, 4])
with _fc1:
    _period = st.selectbox('Filter by period', _PERIODS, index=0, key='dr_period')

_fstart, _fend = None, None
if _period == 'This Week':
    _fstart = _current_week_start.strftime('%Y-%m-%d')
elif _period == 'Last Week':
    _fstart = (_current_week_start - _td(days=7)).strftime('%Y-%m-%d')
    _fend   = (_current_week_start - _td(days=1)).strftime('%Y-%m-%d')
elif _period == 'Last 2 Weeks':
    _fstart = (_today_dt - _td(days=14)).strftime('%Y-%m-%d')
elif _period == 'This Month':
    _fstart = _today_dt.replace(day=1).strftime('%Y-%m-%d')
elif _period == 'Last Month':
    _first_this = _today_dt.replace(day=1)
    _last_prev  = _first_this - _td(days=1)
    _fstart = _last_prev.replace(day=1).strftime('%Y-%m-%d')
    _fend   = _last_prev.strftime('%Y-%m-%d')
elif _period == 'Custom Range':
    with _fc2:
        _cr = st.date_input('Select date range', value=[], key='dr_custom_range')
        if len(_cr) == 2:
            _fstart = _cr[0].strftime('%Y-%m-%d')
            _fend   = _cr[1].strftime('%Y-%m-%d')

# Keep unfiltered copy for pending plays (always show all unresolved regardless of range)
_df_all = df.copy()

if _fstart:
    df = df[df['date_str'] >= _fstart]
if _fend:
    df = df[df['date_str'] <= _fend]

if _period != 'All Time':
    _range_label = f"{_fstart} → {_fend or 'today'}"
    st.caption(f'Showing: **{_period}** ({_range_label})')

# Apply current criteria
today_str = today_str_et()
criteria = df[df['rating'] >= 85]

decided = criteria[criteria['result'].isin(['W', 'L'])]
pending = _df_all[(_df_all['rating'] >= 85) & (_df_all['result'] == '')]

UNIT = 8.0   # dollars per unit
ODDS = -125  # sportsbook odds (American)
_WIN_MULT = 100 / 125  # payout multiplier for -125

def get_units(rating):
    return 1.0

def play_profit(rating, result):
    """Profit in dollars for a single play at -125."""
    u = get_units(rating)
    stake = u * UNIT
    if result == 'W':
        return round(stake * _WIN_MULT, 2)
    return -stake

def play_units_pl(rating, result):
    """Profit in units for a single play at -125."""
    u = get_units(rating)
    if result == 'W':
        return round(u * _WIN_MULT, 3)
    return -u

# ── Staking guide ─────────────────────────────────────────────────────────────

st.markdown('### Staking Guide')
st.caption(f'Flat 1u per play · ${UNIT:.0f}/unit at {ODDS} odds (break-even: 55.6%)')

stake_html = '''<table style="width:100%;border-collapse:collapse;font-family:monospace;">
<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:13px;">
<th style="padding:9px 12px;text-align:left;">Rating</th>
<th style="padding:9px 12px;text-align:center;">Units</th>
<th style="padding:9px 12px;text-align:center;">Bet Amount</th>
<th style="padding:9px 12px;text-align:left;">Notes</th>
</tr></thead><tbody>
<tr style="background:#1a2744;">
  <td style="padding:8px 12px;color:#22c55e;font-weight:700;">75+</td>
  <td style="padding:8px 12px;text-align:center;color:#fbbf24;font-weight:800;">1u</td>
  <td style="padding:8px 12px;text-align:center;color:#e0f2fe;font-weight:700;">$8</td>
  <td style="padding:8px 12px;color:#94a3b8;font-size:12px;">Flat bet — building sample size</td>
</tr>
</tbody></table>'''
st.markdown(stake_html, unsafe_allow_html=True)

st.markdown('---')

# ── Overall summary ───────────────────────────────────────────────────────────

total_w = (decided['result'] == 'W').sum()
total_l = (decided['result'] == 'L').sum()
total_n = len(decided)
total_wr = round(total_w / total_n * 100, 1) if total_n > 0 else None

# Weighted profit and units
total_profit_dollars = sum(
    play_profit(row['rating'], row['result'])
    for _, row in decided.iterrows()
    if pd.notna(row['rating'])
)
total_units_pl = sum(
    play_units_pl(row['rating'], row['result'])
    for _, row in decided.iterrows()
    if pd.notna(row['rating'])
)
total_units_risked = sum(
    get_units(row['rating']) * UNIT
    for _, row in decided.iterrows()
    if pd.notna(row['rating'])
)
total_roi = round(total_profit_dollars / total_units_risked * 100, 1) if total_units_risked > 0 else None

units_str = f'+{total_units_pl:.2f}u' if total_units_pl >= 0 else f'{total_units_pl:.2f}u'

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric('Qualifying Plays', len(criteria))
c2.metric('Decided', total_n)
c3.metric('Record', f'{total_w}-{total_l}' if total_n > 0 else '—')
c4.metric('Win Rate', f'{total_wr}%' if total_wr else '—')
c5.metric('Units', units_str if total_n > 0 else '—')
c6.metric('Net Profit', f'{"+" if total_profit_dollars >= 0 else ""}${total_profit_dollars:.2f}' if total_n > 0 else '—')

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
        day_profit = sum(
            play_profit(row['rating'], row['result'])
            for _, row in grp.iterrows()
            if pd.notna(row['rating'])
        )
        day_units = sum(
            play_units_pl(row['rating'], row['result'])
            for _, row in grp.iterrows()
            if pd.notna(row['rating'])
        )
        daily_rows.append({
            'Date':       date,
            'Record':     f'{w}-{l}',
            'Win Rate':   f'{wr}%',
            '_profit':    day_profit,
            '_units':     day_units,
            'Profit':     f'+${day_profit:.2f}' if day_profit >= 0 else f'-${abs(day_profit):.2f}',
            'Plays':      n,
        })

    daily_df = pd.DataFrame(daily_rows).sort_values('Date', ascending=False)

    def row_color(wr_str):
        wr = float(wr_str.replace('%', ''))
        if wr >= 60:   return '#14532d'
        if wr >= 55.6: return '#1c3a1a'
        if wr > 0:     return '#450a0a'
        return '#1e293b'

    def wr_color(wr_str):
        wr = float(wr_str.replace('%', ''))
        if wr >= 60:   return '#22c55e'
        if wr >= 55.6: return '#eab308'
        return '#ef4444'

    html = '''<table style="width:100%;border-collapse:collapse;font-family:monospace;">
<thead><tr style="background:#1e3a5f;color:#38bdf8;font-size:13px;">
<th style="padding:10px 12px;text-align:left;">Date</th>
<th style="padding:10px 12px;text-align:center;">Record</th>
<th style="padding:10px 12px;text-align:center;">Win Rate</th>
<th style="padding:10px 12px;text-align:center;">Day Units</th>
<th style="padding:10px 12px;text-align:center;">Running Units</th>
<th style="padding:10px 12px;text-align:center;">Profit ($8/unit, -125)</th>
<th style="padding:10px 12px;text-align:center;">Plays</th>
</tr></thead><tbody>'''

    running_profit = 0.0
    running_units  = 0.0
    # Sort ascending to build running total correctly, display newest first
    for _, row in daily_df.sort_values('Date', ascending=True).iterrows():
        bg  = row_color(row['Win Rate'])
        wrc = wr_color(row['Win Rate'])
        p   = row['_profit']
        du  = row['_units']
        running_profit += p
        running_units  += du
        ru_str  = f'+{running_units:.2f}u' if running_units >= 0 else f'{running_units:.2f}u'
        du_str  = f'+{du:.2f}u' if du >= 0 else f'{du:.2f}u'
        ru_color = '#22c55e' if running_units >= 0 else '#ef4444'
        du_color = '#22c55e' if du >= 0 else '#ef4444'
        html += f'''<tr style="background:{bg};border-bottom:1px solid #334155;">
<td style="padding:9px 12px;color:#e0f2fe;">{row['Date']}</td>
<td style="padding:9px 12px;text-align:center;color:#e0f2fe;font-weight:700;">{row['Record']}</td>
<td style="padding:9px 12px;text-align:center;color:{wrc};font-weight:800;">{row['Win Rate']}</td>
<td style="padding:9px 12px;text-align:center;color:{du_color};font-weight:700;">{du_str}</td>
<td style="padding:9px 12px;text-align:center;color:{ru_color};font-weight:800;">{ru_str}</td>
<td style="padding:9px 12px;text-align:center;color:{"#22c55e" if p >= 0 else "#ef4444"};font-weight:700;">{row['Profit']}</td>
<td style="padding:9px 12px;text-align:center;color:#94a3b8;">{row['Plays']}</td>
</tr>'''

    total_ru_str = f'+{total_units_pl:.2f}u' if total_units_pl >= 0 else f'{total_units_pl:.2f}u'
    html += f'''<tr style="background:#0f172a;border-top:2px solid #38bdf8;">
<td style="padding:10px 12px;color:#38bdf8;font-weight:800;">TOTAL</td>
<td style="padding:10px 12px;text-align:center;color:#e0f2fe;font-weight:800;">{total_w}-{total_l}</td>
<td style="padding:10px 12px;text-align:center;color:{"#22c55e" if (total_wr or 0) >= 55.6 else "#ef4444"};font-weight:800;">{total_wr}%</td>
<td style="padding:10px 12px;text-align:center;color:#94a3b8;">—</td>
<td style="padding:10px 12px;text-align:center;color:{"#22c55e" if total_units_pl >= 0 else "#ef4444"};font-weight:800;">{total_ru_str}</td>
<td style="padding:10px 12px;text-align:center;color:{"#22c55e" if total_profit_dollars >= 0 else "#ef4444"};font-weight:800;">{"+" if total_profit_dollars >= 0 else ""}${total_profit_dollars:.2f}</td>
<td style="padding:10px 12px;text-align:center;color:#94a3b8;">{total_n}</td>
</tr></tbody></table>'''

    st.markdown(html, unsafe_allow_html=True)

    # Per-play breakdown per day
    st.markdown('#### Per-Play Breakdown')
    for date, grp in decided.groupby('date_str', sort=False):
        grp = grp.sort_values('rating', ascending=False)
        with st.expander(f'📆 {date}', expanded=False):
            play_html = '''<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;">
<thead><tr style="background:#1e3a5f;color:#38bdf8;">
<th style="padding:7px 10px;text-align:left;">Player</th>
<th style="padding:7px 10px;text-align:center;">Rating</th>
<th style="padding:7px 10px;text-align:center;">Proj</th>
<th style="padding:7px 10px;text-align:center;">Units</th>
<th style="padding:7px 10px;text-align:center;">Bet</th>
<th style="padding:7px 10px;text-align:center;">Result</th>
<th style="padding:7px 10px;text-align:center;">P/L</th>
</tr></thead><tbody>'''
            for _, play in grp.iterrows():
                r = int(play['rating']) if pd.notna(play['rating']) else 75
                u = get_units(r)
                p = play_profit(r, play['result'])
                res_color = '#22c55e' if play['result'] == 'W' else '#ef4444'
                pl_color  = '#22c55e' if p >= 0 else '#ef4444'
                pl_str    = f'+${p:.2f}' if p >= 0 else f'-${abs(p):.2f}'
                play_html += f'''<tr style="border-bottom:1px solid #1e293b;">
<td style="padding:7px 10px;color:#e0f2fe;">{play.get("player","—")}</td>
<td style="padding:7px 10px;text-align:center;color:#7dd3fc;">{r}</td>
<td style="padding:7px 10px;text-align:center;color:#94a3b8;">{play["projected"] if pd.notna(play["projected"]) else "—"}</td>
<td style="padding:7px 10px;text-align:center;color:#fbbf24;font-weight:700;">{u}u</td>
<td style="padding:7px 10px;text-align:center;color:#e0f2fe;">${u * UNIT:.0f}</td>
<td style="padding:7px 10px;text-align:center;color:{res_color};font-weight:800;">{play["result"]}</td>
<td style="padding:7px 10px;text-align:center;color:{pl_color};font-weight:700;">{pl_str}</td>
</tr>'''
            play_html += '</tbody></table>'
            st.markdown(play_html, unsafe_allow_html=True)

st.markdown('---')

# ── Pending plays ─────────────────────────────────────────────────────────────

if not pending.empty:
    st.markdown(f'### Pending Plays ({len(pending)})')
    st.caption("Today's qualifying plays — results not in yet.")

    pend_rows = []
    for _, row in pending.sort_values('date_str', ascending=False).iterrows():
        r = int(row['rating']) if pd.notna(row['rating']) else None
        u = get_units(r) if r else 1.0
        pend_rows.append({
            'Date':      row['date_str'],
            'Player':    row.get('player', ''),
            'Team':      row.get('team', ''),
            'Rating':    r if r else '—',
            'Projected': row['projected'] if pd.notna(row['projected']) else '—',
            'Line':      row.get('line', '—'),
            'Units':     f'{u}u',
            'Bet':       f'${u * UNIT:.0f}',
        })

    st.dataframe(pd.DataFrame(pend_rows), hide_index=True, use_container_width=True)

st.markdown('---')

# ── Needs a line to grade — past plays that have an actual but no line ─────────
# These are the "yesterday won't load in" plays: the result is in, but no line
# was recorded (lines weren't pulling), so they can't be scored W/L and stay
# out of the day-by-day record. Enter the line here to grade them instantly.
_gl = _df_all.copy()
_gl['_ds'] = _gl['date'].astype(str).str[:10]
_gl['_a']  = _gl['actual'].astype(str).str.strip()
_gl['_r']  = _gl['result'].astype(str).str.strip()
_gl['_l']  = _gl['line'].astype(str).str.strip() if 'line' in _gl.columns else ''
_gl['_rn'] = pd.to_numeric(_gl['rating'], errors='coerce')
_needs_line = _gl[
    (_gl['_ds'] < today_str_et()) &
    (_gl['_rn'] >= 85) &
    (~_gl['_a'].isin(['', 'nan'])) &
    (_gl['_r'].isin(['', 'nan'])) &
    (_gl['_l'].isin(['', 'nan']))
]

if not _needs_line.empty:
    st.markdown(f'### ⚠️ Needs a Line to Grade ({len(_needs_line)})')
    st.caption('These plays have a final H+R+RBI but no line was recorded, so they '
               "can't be scored. Enter each line and save to grade them and pull "
               'them into the record.')

    _edit_src = _needs_line[['_ds', 'player', 'team', 'rating', 'projected', 'actual']].copy()
    _edit_src = _edit_src.rename(columns={'_ds': 'Date', 'player': 'Player',
                                          'team': 'Team', 'rating': 'Rating',
                                          'projected': 'Proj', 'actual': 'Actual'})
    _edit_src['Line'] = None
    _edit_src = _edit_src.sort_values('Date', ascending=False).reset_index(drop=True)

    _edited = st.data_editor(
        _edit_src, hide_index=True, use_container_width=True, key='dr_grade_editor',
        column_config={
            'Date':   st.column_config.TextColumn('Date',   disabled=True),
            'Player': st.column_config.TextColumn('Player', disabled=True),
            'Team':   st.column_config.TextColumn('Team',   disabled=True),
            'Rating': st.column_config.NumberColumn('Rating', disabled=True),
            'Proj':   st.column_config.NumberColumn('Proj',   disabled=True),
            'Actual': st.column_config.NumberColumn('Actual', disabled=True),
            'Line':   st.column_config.NumberColumn('Line', min_value=0.0,
                                                    max_value=10.0, step=0.5,
                                                    help='Enter the sportsbook line'),
        },
    )

    if st.button('✅ Save Lines & Grade', type='primary', key='dr_grade_save'):
        _full = load_all()
        _full['_ds'] = _full['date'].astype(str).str[:10]
        _graded = 0
        for _, _er in _edited.iterrows():
            if _er['Line'] is None or pd.isna(_er['Line']):
                continue
            _mask = (_full['_ds'] == _er['Date']) & (_full['player'] == _er['Player'])
            for _idx in _full[_mask].index:
                _full.at[_idx, 'line'] = str(_er['Line'])
                try:
                    _av = float(str(_full.at[_idx, 'actual']).strip())
                    _full.at[_idx, 'result'] = 'W' if _av > float(_er['Line']) else 'L'
                    _graded += 1
                except (ValueError, TypeError):
                    pass
        _full.drop(columns=['_ds'], inplace=True, errors='ignore')
        save_all(_full)
        st.success(f'Graded {_graded} play(s)!')
        st.rerun()

    st.markdown('---')

# ── Correct a pending play ────────────────────────────────────────────────────

with st.expander('✏️ Correct a Pending Play', expanded=False):
    st.caption('Enter the actual HRR for a pending play to mark it W or L.')
    if pending.empty:
        st.info('No pending plays to correct.')
    else:
        _pend_dates = sorted(pending['date_str'].unique(), reverse=True)
        _corr_date = st.selectbox('Date', _pend_dates, key='dr_corr_date')
        _day_pend = pending[pending['date_str'] == _corr_date]
        _corr_player = st.selectbox('Player', _day_pend['player'].tolist(), key='dr_corr_player')

        _corr_row = _day_pend[_day_pend['player'] == _corr_player].iloc[0]
        _proj = _corr_row['projected'] if pd.notna(_corr_row['projected']) else '—'
        _line_raw = _corr_row.get('line', None)
        try:
            _line_val = float(_line_raw) if _line_raw not in (None, '', 'nan', '—') else 1.5
        except (ValueError, TypeError):
            _line_val = 1.5

        cc1, cc2 = st.columns(2)
        cc1.metric('Projected', str(_proj))
        cc2.metric('Line', str(_line_val))

        _actual_val = st.number_input('Actual HRR', min_value=0, max_value=30, value=0, step=1, key='dr_corr_actual')
        _auto_result = 'W' if _actual_val > _line_val else 'L'
        st.info(f'Result will be marked: **{_auto_result}** (actual {_actual_val} vs line {_line_val})')

        if st.button('✅ Save Correction', type='primary', key='dr_corr_btn'):
            _cdf = load_all()
            _mask = (
                (_cdf['player'] == _corr_player) &
                (_cdf['date'].astype(str).str[:10] == _corr_date)
            )
            if _mask.any():
                _idx = _cdf[_mask].index[0]
                _cdf.at[_idx, 'actual'] = str(_actual_val)
                _cdf.at[_idx, 'result'] = _auto_result
                save_all(_cdf)
                st.success(f'Updated {_corr_player} ({_corr_date}): actual={_actual_val}, result={_auto_result}')
                st.rerun()
            else:
                st.error('Play not found in log.')

st.markdown('---')

# ── Manual add ────────────────────────────────────────────────────────────────

with st.expander('➕ Manually Add a Play', expanded=False):
    st.caption('Add a play directly to the log (bypasses Game View). Use for players missed by auto-logging.')
    with st.form('manual_add_dr'):
        mc1, mc2 = st.columns(2)
        m_player  = mc1.text_input('Player Name', placeholder='e.g. James Wood')
        m_team    = mc2.text_input('Team', placeholder='e.g. WSH')
        mc3, mc4, mc5 = st.columns(3)
        m_rating  = mc3.number_input('Rating', min_value=0, max_value=100, value=80, step=1)
        m_proj    = mc4.number_input('Projected HRR', min_value=0.0, max_value=20.0, value=1.5, step=0.1)
        m_line    = mc5.number_input('Line', min_value=0.5, max_value=10.0, value=1.5, step=0.5)
        mc6, mc7 = st.columns(2)
        m_date    = mc6.date_input('Game Date', value=today_et())
        m_pitcher = mc7.text_input('vs Pitcher', placeholder='optional')
        m_actual  = st.number_input('Actual HRR (leave -1 if still pending)', min_value=-1, max_value=30, value=-1, step=1)
        submitted = st.form_submit_button('Add Play', type='primary', use_container_width=True)

    if submitted:
        if not m_player.strip():
            st.error('Player name is required.')
        else:
            game_date = m_date.strftime('%Y-%m-%d')
            _grade = 'A+' if m_rating >= 85 else 'A' if m_rating >= 80 else 'B+'
            log_play(
                player=m_player.strip(),
                team=m_team.strip(),
                rating=int(m_rating),
                grade=_grade,
                projected=float(m_proj),
                line=float(m_line),
                vs_pitcher=m_pitcher.strip(),
                game_date=game_date,
                game_started=False,
            )
            if m_actual >= 0:
                _df = load_all()
                mask = (
                    (_df['player'] == m_player.strip()) &
                    (_df['date'].astype(str).str[:10] == game_date)
                )
                if mask.any():
                    idx = _df[mask].index[0]
                    _df.at[idx, 'actual'] = str(m_actual)
                    if game_date < today_str_et():
                        _df.at[idx, 'result'] = 'W' if m_actual > float(m_line) else 'L'
                    save_all(_df)
            st.success(f'Added {m_player.strip()} ({game_date})!')
            st.rerun()

st.markdown('---')

# ── Remove a play ─────────────────────────────────────────────────────────────

with st.expander('🗑️ Remove a Play', expanded=False):
    st.caption('Permanently removes a play from the full play log (Daily Results & Analytics).')
    _all_dates = sorted(df['date_str'].unique(), reverse=True)
    if _all_dates:
        _rem_date   = st.selectbox('Date', _all_dates, key='dr_rem_date')
        _day_plays  = df[df['date_str'] == _rem_date]['player'].tolist()
        if _day_plays:
            _rem_player = st.selectbox('Player', _day_plays, key='dr_rem_player')
            st.warning(f'This will permanently delete **{_rem_player}** ({_rem_date}) from the play log.')
            if st.button('🗑️ Remove Play', type='primary', key='dr_rem_btn'):
                _rdf = load_all()
                _rdf = _rdf[~((_rdf['player'] == _rem_player) & (_rdf['date'].astype(str).str[:10] == _rem_date))].reset_index(drop=True)
                save_all(_rdf)
                st.success(f'Removed {_rem_player} ({_rem_date}).')
                st.rerun()
        else:
            st.info('No plays on this date.')

st.markdown('---')

# ── Play log for current criteria ─────────────────────────────────────────────

with st.expander('📋 Full Play Log (current criteria)', expanded=False):
    show_cols = ['date_str', 'player', 'team', 'rating', 'projected', 'line', 'actual', 'result']
    show_cols = [c for c in show_cols if c in criteria.columns]
    display = criteria[show_cols].rename(columns={'date_str': 'date'}).sort_values('date', ascending=False)
    st.dataframe(display, hide_index=True, use_container_width=True)
