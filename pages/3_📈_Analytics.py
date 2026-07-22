"""
Analytics — tracks every HRR play and finds the most profitable
rating ranges, projection ranges, and key number combinations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from full_tracker import load_all, update_actuals, save_all
from shared_styles import inject_styles

st.set_page_config(page_title="Analytics | MLB Props", page_icon="📈", layout="wide")
inject_styles()


def win_rate(df):
    d = df[df['result'].isin(['W','L'])]
    if len(d) == 0:
        return None, 0
    w = (d['result'] == 'W').sum()
    return round(w / len(d) * 100, 1), len(d)


def roi(df):
    """Simple ROI assuming -110 standard vig."""
    decided = df[df['result'].isin(['W','L'])]
    if len(decided) == 0:
        return None
    wins   = (decided['result'] == 'W').sum()
    losses = (decided['result'] == 'L').sum()
    # Assume -110 (bet 110 to win 100)
    profit = wins * 100 - losses * 110
    return round(profit / (len(decided) * 110) * 100, 1)


def color_wr(wr):
    if wr is None:
        return '#475569'
    if wr >= 60:
        return '#22c55e'
    if wr >= 52:
        return '#eab308'
    return '#ef4444'


st.markdown('## 📈 Play Analytics')
st.caption('Auto-updates after every game. Tracks every HRR play to find the most profitable thresholds.')

# Show play log size for debugging
_total_plays = len(load_all())
if _total_plays == 0:
    st.warning('⚠️ No plays logged yet. Open the **Game View** page and let lineups fully load — every batter gets logged automatically.')
else:
    st.caption(f'📊 {_total_plays} plays in log')

# Clear any actuals/results for today's games (may have been fetched mid-game)
from datetime import datetime as _datetime
from eastern_time import today_str_et
from full_tracker import load_all as _load_all, save_all as _save_all
_today_str = today_str_et()
_full_df = _load_all()
if not _full_df.empty:
    _today_mask = _full_df['date'].astype(str).str[:10] >= _today_str
    if _today_mask.any():
        _full_df = _full_df.copy()
        _full_df.loc[_today_mask, 'actual'] = ''
        _full_df.loc[_today_mask, 'result'] = ''
        _save_all(_full_df)

# ── Controls ──────────────────────────────────────────────────────────────────

col_refresh, col_fetch, col_sync = st.columns([1, 1, 1])
with col_refresh:
    if st.button('🔄 Refresh', use_container_width=True):
        st.session_state.pop('analytics_last_update', None)
        st.rerun()
with col_sync:
    if st.button('🔁 Sync from Tracker', use_container_width=True,
                 help='Copy any missing plays from the betting tracker into the analytics log'):
        from tracker import load as load_tracker
        tracker_df = load_tracker()
        analytics_df = load_all()
        added = 0
        for _, row in tracker_df.iterrows():
            player = str(row.get('player', ''))
            date   = str(row.get('date', ''))[:10]
            if player and date:
                exists = (not analytics_df.empty and
                         ((analytics_df['player'] == player) &
                          (analytics_df['date'].astype(str).str[:10] == date)).any())
                if not exists:
                    from full_tracker import log_play
                    try:
                        log_play(
                            player=player, team=str(row.get('team','')),
                            rating=int(float(row.get('rating', 0) or 0)),
                            grade=str(row.get('grade','')),
                            projected=float(row.get('projected', 0) or 0),
                            line=float(row.get('line', 0) or 0) if row.get('line') and str(row.get('line')) not in ('', 'nan') else None,
                            vs_pitcher=str(row.get('vs_pitcher','')),
                            game_date=date
                        )
                        added += 1
                    except Exception:
                        pass
        st.success(f'Synced {added} missing play(s) from tracker!')
        st.rerun()

with col_fetch:
    if st.button('⬇️ Force Fetch All Actuals', type='primary', use_container_width=True):
        st.session_state.pop('analytics_last_update', None)
        with st.spinner('Fetching all results from MLB API...'):
            n = update_actuals()
        st.success(f'Updated {n} plays!')
        st.rerun()

df = load_all()

if df.empty:
    st.info('No play data yet. Open the Game View page to start logging plays automatically.')
    st.stop()

# ── Manual correction ─────────────────────────────────────────────────────────
with st.expander('✏️ Manually Correct an Actual', expanded=False):
    st.caption('Fix wrong actuals caused by mid-game fetches.')
    dates_a  = sorted(df['date'].astype(str).str[:10].unique(), reverse=True) if not df.empty else []
    sel_date_a = st.selectbox('Date', dates_a, key='analytics_manual_date')
    day_df_a   = df[df['date'].astype(str).str[:10] == sel_date_a] if sel_date_a else df
    players_a  = day_df_a['player'].tolist()
    sel_a = st.selectbox('Player', players_a, key='analytics_manual_player')
    new_a = st.number_input('Correct Actual H+R+RBI', min_value=0, max_value=20, step=1, key='analytics_manual_actual')
    if st.button('✅ Apply', type='primary', key='analytics_apply'):
        idx = df[(df['player'] == sel_a) & (df['date'].astype(str).str[:10] == sel_date_a)].index
        if len(idx) > 0:
            df = df.copy()
            i = idx[0]
            df.at[i, 'actual'] = str(new_a)
            line_val = str(df.at[i, 'line']).strip()
            line = float(line_val) if line_val and line_val not in ('nan', '') else 1.5
            today_s = today_str_et()
            df.at[i, 'result'] = 'W' if (new_a > line and sel_date_a < today_s) else 'L' if sel_date_a < today_s else ''
            save_all(df)
            st.success(f'Updated {sel_a} ({sel_date_a}): actual={new_a}')
            st.rerun()

# Ensure numeric types
df['rating']    = pd.to_numeric(df['rating'],    errors='coerce')
df['projected'] = pd.to_numeric(df['projected'], errors='coerce')
df['line']      = pd.to_numeric(df['line'],      errors='coerce')
df['actual']    = pd.to_numeric(df['actual'],    errors='coerce')
df['date_str']  = df['date'].astype(str).str[:10]

# ── Benchmark grading (ANALYTICS VIEW ONLY — never saved) ─────────────────────
# The betting record (Tracker / Daily Results) requires a real sportsbook line.
# But this log records every batter and most never get a line, so for the band
# win-rate research below, ungraded past plays are scored against the real line
# when one exists, else a fixed 1.5 benchmark. This derived result exists only
# in this page's dataframe — save/download paths reload from the database.
_ungraded = ((df['result'].astype(str).str.strip() == '') &
             df['actual'].notna() & (df['date_str'] < _today_str))
_with_line = _ungraded & df['line'].notna()
_no_line   = _ungraded & df['line'].isna()
df.loc[_with_line, 'result'] = np.where(df.loc[_with_line, 'actual'] > df.loc[_with_line, 'line'], 'W', 'L')
df.loc[_no_line,   'result'] = np.where(df.loc[_no_line,   'actual'] > 1.5, 'W', 'L')
if int(_no_line.sum()):
    st.caption(f'ℹ️ {int(_no_line.sum())} play(s) without a recorded line are scored '
               f'against a **1.5 benchmark** here for research — they do not count '
               f'in the Tracker or Daily Results.')

# ── Date filter ───────────────────────────────────────────────────────────────

from datetime import datetime as _dt, timedelta as _td

_today_dt           = _dt.now().date()
_current_week_start = _today_dt - _td(days=_today_dt.weekday())

_PERIODS = ['All Time', 'This Week', 'Last Week', 'Last 2 Weeks',
            'This Month', 'Last Month', 'Custom Range']

_fc1, _fc2 = st.columns([2, 4])
with _fc1:
    _period = st.selectbox('Filter by period', _PERIODS, index=0)

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
        _cr = st.date_input('Select date range', value=[], key='analytics_custom_range')
        if len(_cr) == 2:
            _fstart = _cr[0].strftime('%Y-%m-%d')
            _fend   = _cr[1].strftime('%Y-%m-%d')

if _fstart:
    df = df[df['date_str'] >= _fstart]
if _fend:
    df = df[df['date_str'] <= _fend]

if _period != 'All Time':
    _range_label = f"{_fstart} → {_fend or 'today'}"
    st.caption(f'Showing: **{_period}** ({_range_label}) — {len(df)} plays')

decided = df[df['result'].isin(['W','L'])]

# ── Rating band diagnostic ────────────────────────────────────────────────────
# Inspect the plays behind a band to see WHY it wins or loses: is the projection
# inflated for this band, are the lines too high, or is it just variance?
with st.expander('🔬 Rating Band Diagnostic — why a band wins or loses', expanded=False):
    st.caption('Respects the date filter above. The key tell is Avg Actual vs Avg '
               'Projection: a big negative gap means the model over-projects this band '
               '(inflated ratings → fake edge → losses).')
    _bands = {'95+': (95, 101), '90-94': (90, 95), '85-89': (85, 90),
              '80-84': (80, 85), '75-79': (75, 80), '70-74': (70, 75),
              '65-69': (65, 70), '60-64': (60, 65)}
    _sb = st.selectbox('Rating band', list(_bands.keys()), index=1, key='band_diag_sel')
    _blo, _bhi = _bands[_sb]
    _bd = df[(df['rating'] >= _blo) & (df['rating'] < _bhi)].copy()
    _bd_dec = _bd[_bd['result'].isin(['W', 'L'])]
    if _bd_dec.empty:
        st.info('No decided plays in this band for the selected period.')
    else:
        _w  = int((_bd_dec['result'] == 'W').sum())
        _l  = int((_bd_dec['result'] == 'L').sum())
        _wr = round(_w / (_w + _l) * 100, 1)
        _avg_proj = _bd_dec['projected'].mean()
        _avg_act  = _bd_dec['actual'].mean()
        _avg_line = _bd_dec['line'].mean()
        _bias     = _avg_act - _avg_proj
        _n_line   = int(_bd_dec['line'].notna().sum())

        d1, d2, d3, d4 = st.columns(4)
        d1.metric('Record', f'{_w}-{_l}', f'{_wr}%', delta_color='off')
        d2.metric('Avg Projection', f'{_avg_proj:.2f}')
        d3.metric('Avg Actual', f'{_avg_act:.2f}', f'{_bias:+.2f} vs proj',
                  delta_color='normal' if _bias >= 0 else 'inverse')
        d4.metric('Avg Line', f'{_avg_line:.2f}' if pd.notna(_avg_line) else '—',
                  f'{_n_line}/{_w + _l} real lines', delta_color='off')

        _wins   = _bd_dec[_bd_dec['result'] == 'W']
        _losses = _bd_dec[_bd_dec['result'] == 'L']
        st.caption(
            f"**Wins** ({_w}): avg proj {_wins['projected'].mean():.2f} · avg actual {_wins['actual'].mean():.2f}  |  "
            f"**Losses** ({_l}): avg proj {_losses['projected'].mean():.2f} · avg actual {_losses['actual'].mean():.2f}"
        )

        _show = _bd_dec[['date_str', 'player', 'rating', 'projected', 'line', 'actual', 'result', 'vs_pitcher']].copy()
        _show['edge'] = (_show['projected'] - _show['line']).round(2)
        _show = _show.sort_values('date_str', ascending=False)
        st.dataframe(_show, hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False, ttl=3600)
def _recon_player_hrr(name: str, season: int):
    """Pull a player's real game-by-game HRR from the MLB API (cached)."""
    try:
        import statsapi as _sa
        from data_collector import get_game_logs
        matches = _sa.lookup_player(name)
        if not matches:
            return None
        g = get_game_logs(int(matches[0]['id']), [season])
        if g.empty:
            return None
        g = g[['date', 'h', 'r', 'rbi']].copy()
        g['hrr'] = g['h'] + g['r'] + g['rbi']
        return g[['date', 'hrr']]
    except Exception:
        return None


# ── boom_delta reconstruction ─────────────────────────────────────────────────
# Pull each player's REAL game logs (MLB API) to compute a clean trailing HRR
# baseline, then boom_delta = projected - baseline. Reveals whether over-
# projection separates wins from losses so we can fit the penalty threshold.
with st.expander('🔧 boom_delta Reconstruction (API pull) — find the win/loss knee', expanded=False):
    st.caption("Rebuilds each play's baseline from the player's real game logs, so it's "
               "immune to the polluted actuals in the log. boom_delta = projected − trailing "
               "HRR. If losses cluster at higher boom_delta than wins, over-projection is the "
               "separator — set the penalty threshold near the split.")
    _rbands = {'95+': (95, 101), '90-94': (90, 95), '85-89': (85, 90),
               '80-84': (80, 85), '75-79': (75, 80), '70-74': (70, 75)}
    _rc1, _rc2 = st.columns([2, 2])
    with _rc1:
        _rband = st.selectbox('Rating band', list(_rbands.keys()), index=1, key='boom_recon_band')
    with _rc2:
        _winN = st.slider('Baseline window (games)', 10, 30, 20, key='boom_recon_win')
    _rlo, _rhi = _rbands[_rband]
    _rset = df[(df['rating'] >= _rlo) & (df['rating'] < _rhi) &
               df['result'].isin(['W', 'L'])].copy()
    _nplayers = _rset['player'].nunique()
    st.caption(f'{len(_rset)} decided plays · {_nplayers} unique players to pull'
               + (' — this may take a moment' if _nplayers > 30 else ''))
    if st.button('Run reconstruction', key='boom_recon_run', type='primary'):
        if _rset.empty:
            st.info('No decided plays in this band for the selected period.')
        else:
            _logs, _prog = {}, st.progress(0.0)
            _players = list(_rset['player'].unique())
            for _i, _pname in enumerate(_players):
                _season = int(str(_rset[_rset['player'] == _pname]['date_str'].iloc[0])[:4])
                _logs[_pname] = _recon_player_hrr(_pname, _season)
                _prog.progress((_i + 1) / len(_players))
            _prog.empty()
            _rows = []
            for _, _r in _rset.iterrows():
                _g = _logs.get(_r['player'])
                _proj = _r['projected']
                if _g is None or _g.empty or pd.isna(_proj):
                    continue
                _pdate = pd.to_datetime(_r['date_str'])
                _prior = _g[_g['date'] < _pdate].tail(_winN)
                if len(_prior) < 10:
                    continue
                _base = _prior['hrr'].mean()
                _rows.append({'date': _r['date_str'], 'player': _r['player'],
                              'rating': int(_r['rating']), 'projected': round(float(_proj), 2),
                              'baseline': round(float(_base), 2),
                              'boom_delta': round(float(_proj) - float(_base), 2),
                              'result': _r['result']})
            if not _rows:
                st.warning('Not enough game-log history (need ≥10 prior games per play) to reconstruct.')
            else:
                _rec = pd.DataFrame(_rows)
                _wd = _rec[_rec['result'] == 'W']['boom_delta']
                _ld = _rec[_rec['result'] == 'L']['boom_delta']
                m1, m2, m3 = st.columns(3)
                m1.metric('Wins — avg boom_delta',
                          f'{_wd.mean():+.2f}' if not _wd.empty else '—',
                          f'{len(_wd)} plays', delta_color='off')
                m2.metric('Losses — avg boom_delta',
                          f'{_ld.mean():+.2f}' if not _ld.empty else '—',
                          f'{len(_ld)} plays', delta_color='off')
                _sep = (_ld.mean() - _wd.mean()) if (not _wd.empty and not _ld.empty) else 0.0
                m3.metric('Separation (L − W)', f'{_sep:+.2f}',
                          'losses more inflated' if _sep > 0.3 else 'weak split', delta_color='off')
                st.caption(f'{len(_rec)} of {len(_rset)} plays had enough history. '
                           'Sorted by boom_delta — scan where W flips to L.')
                st.dataframe(_rec.sort_values('boom_delta', ascending=False),
                             hide_index=True, use_container_width=True)

# ── Overall record ────────────────────────────────────────────────────────────

st.markdown('---')
total_wr, total_n = win_rate(df)
total_roi         = roi(df)

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric('Total Plays',  len(df))
c2.metric('Decided',      total_n)
c3.metric('Win Rate',     f'{total_wr}%' if total_wr else '—')
c4.metric('ROI',          f'{total_roi}%' if total_roi else '—')
c5.metric('Pending',      len(df[df['result'] == '']))

st.markdown('---')

# ── Rating buckets (5-point bands) ───────────────────────────────────────────

st.markdown('### Win Rate by Rating')

rating_buckets = [
    (60, 65, '60-64'), (65, 70, '65-69'), (70, 75, '70-74'),
    (75, 80, '75-79'), (80, 85, '80-84'), (85, 90, '85-89'),
    (90, 95, '90-94'), (95, 101, '95+'),
]
r_labels, r_wrs, r_ns, r_rois, r_colors = [], [], [], [], []

for lo, hi, label in rating_buckets:
    sub = df[(df['rating'] >= lo) & (df['rating'] < hi)]
    wr, n = win_rate(sub)
    r_labels.append(label)
    r_wrs.append(wr or 0)
    r_ns.append(n)
    r_rois.append(roi(sub))
    r_colors.append(color_wr(wr))

fig_r = go.Figure(go.Bar(
    x=r_labels, y=r_wrs,
    marker_color=r_colors,
    text=[f'{w}%<br>({n} plays)' if n > 0 else '0 plays'
          for w, n in zip(r_wrs, r_ns)],
    textposition='outside',
))
fig_r.add_hline(y=52.4, line_dash='dash', line_color='#ef4444',
                annotation_text='Break-even (-110)', annotation_position='right')
fig_r.update_layout(
    height=350, yaxis=dict(range=[0, 100], title='Win %'),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7dd3fc'), margin=dict(t=30, b=10),
    xaxis=dict(title='Rating Band'),
)
st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False})

# Rating table
r_df = pd.DataFrame({
    'Rating Band': r_labels,
    'Plays':       r_ns,
    'Win Rate':    [f'{w}%' if n > 0 else '—' for w, n in zip(r_wrs, r_ns)],
    'ROI':         [f'{r}%' if r is not None else '—' for r in r_rois],
})
st.dataframe(r_df, hide_index=True, use_container_width=True)

st.markdown('---')

# ── Rating band × Projection breakdown ───────────────────────────────────────

st.markdown('### Win Rate by Rating Band + Projection Filter')
st.caption('Shows how each 5-point rating band performs at different projection thresholds — find the exact cut that works.')

proj_thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
band_rows = []

for lo, hi, label in rating_buckets:
    band = df[(df['rating'] >= lo) & (df['rating'] < hi)]
    row = {'Rating Band': label, 'Total Plays': len(band[band['result'].isin(['W','L'])])}
    for pt in proj_thresholds:
        sub = band[band['projected'] >= pt]
        wr, n = win_rate(sub)
        w = int((sub[sub['result'].isin(['W','L'])]['result'] == 'W').sum())
        l = int((sub[sub['result'].isin(['W','L'])]['result'] == 'L').sum())
        if n > 0:
            row[f'Proj ≥{pt}'] = f"{wr}% ({w}-{l})"
        else:
            row[f'Proj ≥{pt}'] = '—'
    band_rows.append(row)

band_breakdown = pd.DataFrame(band_rows)
st.dataframe(band_breakdown, hide_index=True, use_container_width=True)

st.markdown('---')

# ── Win Rate by Projection Threshold (All Ratings) ───────────────────────────

st.markdown('### Win Rate by Projection (All Ratings)')
st.caption('Win rate at each projection cutoff across the entire play log, regardless of rating — find the projection floor that works on its own.')

all_proj_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
ap_labels, ap_wrs, ap_ns, ap_rois, ap_colors, ap_wl = [], [], [], [], [], []

for pt in all_proj_thresholds:
    sub = df[df['projected'] >= pt]
    decided_sub = sub[sub['result'].isin(['W','L'])]
    wr, n = win_rate(sub)
    w = int((decided_sub['result'] == 'W').sum())
    l = int((decided_sub['result'] == 'L').sum())
    ap_labels.append(f'≥{pt}')
    ap_wrs.append(wr or 0)
    ap_ns.append(n)
    ap_rois.append(roi(sub))
    ap_colors.append(color_wr(wr))
    ap_wl.append((w, l))

fig_ap = go.Figure(go.Bar(
    x=ap_labels, y=ap_wrs,
    marker_color=ap_colors,
    text=[f'{w}%<br>({n} plays)' if n > 0 else '0 plays'
          for w, n in zip(ap_wrs, ap_ns)],
    textposition='outside',
))
fig_ap.add_hline(y=52.4, line_dash='dash', line_color='#ef4444',
                 annotation_text='Break-even (-110)', annotation_position='right')
fig_ap.update_layout(
    height=350, yaxis=dict(range=[0, 100], title='Win %'),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7dd3fc'), margin=dict(t=30, b=10),
    xaxis=dict(title='Projection Threshold'),
)
st.plotly_chart(fig_ap, use_container_width=True, config={'displayModeBar': False})

ap_df = pd.DataFrame({
    'Filter':   [f'Proj ≥{pt}' for pt in all_proj_thresholds],
    'Plays':    ap_ns,
    'Win Rate': [f'{wr}% ({w}-{l})' if n > 0 else '—'
                 for wr, n, (w, l) in zip(ap_wrs, ap_ns, ap_wl)],
    'ROI':      [f'{r}%' if r is not None else '—' for r in ap_rois],
})
st.dataframe(ap_df, hide_index=True, use_container_width=True)

st.markdown('---')

# ── Projection buckets ────────────────────────────────────────────────────────

st.markdown('### Win Rate by Projection (vs Line)')

# Group by proj - line edge
if df['line'].notna().any():
    df['edge'] = df['projected'] - df['line']
    edge_buckets = [(-99,-1.0,'<-1.0'),(-1.0,-0.5,'-1.0 to -0.5'),
                    (-0.5,0.0,'-0.5 to 0'),( 0.0,0.5,'0 to +0.5'),
                    (0.5,1.0,'+0.5 to +1.0'),(1.0,99,'+1.0+')]
    e_labels,e_wrs,e_ns,e_rois,e_colors = [],[],[],[],[]
    for lo,hi,label in edge_buckets:
        sub = df[(df['edge'] >= lo) & (df['edge'] < hi)]
        wr,n = win_rate(sub)
        e_labels.append(label); e_wrs.append(wr or 0)
        e_ns.append(n); e_rois.append(roi(sub)); e_colors.append(color_wr(wr))

    fig_e = go.Figure(go.Bar(
        x=e_labels, y=e_wrs, marker_color=e_colors,
        text=[f'{w}%<br>({n})' if n>0 else '0' for w,n in zip(e_wrs,e_ns)],
        textposition='outside',
    ))
    fig_e.add_hline(y=52.4, line_dash='dash', line_color='#ef4444')
    fig_e.update_layout(
        height=350, yaxis=dict(range=[0,100], title='Win %'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#7dd3fc'), margin=dict(t=30,b=10),
        xaxis=dict(title='Projection − Line Edge'),
    )
    st.plotly_chart(fig_e, use_container_width=True, config={'displayModeBar': False})
else:
    st.info('Enter sportsbook lines in the Game View to see edge analysis.')

st.markdown('---')

# ── Key number finder ─────────────────────────────────────────────────────────

st.markdown('### Key Number Finder — Most Profitable Thresholds')
st.caption('Minimum rating and projection that maximize win rate with at least 10 plays.')

if len(decided) >= 10:
    results = []
    for min_r in range(40, 85, 5):
        for min_p in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
            sub = df[(df['rating'] >= min_r) & (df['projected'] >= min_p)]
            wr, n = win_rate(sub)
            r_val = roi(sub)
            if n >= 10 and wr is not None:
                results.append({
                    'Min Rating': min_r,
                    'Min Proj':   min_p,
                    'Plays':      n,
                    'Win Rate':   wr,
                    'ROI':        r_val or 0,
                })

    if results:
        res_df = pd.DataFrame(results).sort_values('Win Rate', ascending=False)
        res_df['Win Rate'] = res_df['Win Rate'].apply(lambda x: f'{x}%')
        res_df['ROI']      = res_df['ROI'].apply(lambda x: f'{x}%')
        st.dataframe(res_df.head(20), hide_index=True, use_container_width=True)

        # Highlight the best combo
        best = results[0] if results else None
        if best:
            st.success(
                f"🏆 **Best combo:** Rating ≥ {best['Min Rating']} + Projection ≥ {best['Min Proj']} "
                f"→ **{best['Win Rate']}% win rate** over {best['Plays']} plays"
            )
    else:
        st.info('Not enough data yet. Need at least 10 decided plays per combination.')
else:
    st.info(f'Need at least 10 decided plays for key number analysis. Currently have {len(decided)}.')

st.markdown('---')

# ── Best players ──────────────────────────────────────────────────────────────

st.markdown('### Most Profitable Players (min 5 plays)')

if len(decided) >= 5:
    player_stats = []
    for player, grp in decided.groupby('player'):
        wr, n = win_rate(grp)
        if n >= 5:
            player_stats.append({
                'Player':   player,
                'Plays':    n,
                'W':        int((grp['result']=='W').sum()),
                'L':        int((grp['result']=='L').sum()),
                'Win Rate': f'{wr}%',
                'ROI':      f'{roi(grp)}%' if roi(grp) is not None else '—',
                'Avg Proj': round(grp['projected'].mean(), 2),
                'Avg Rating': round(grp['rating'].mean(), 0),
            })
    if player_stats:
        p_df = pd.DataFrame(player_stats).sort_values('Win Rate', ascending=False)
        st.dataframe(p_df, hide_index=True, use_container_width=True)

st.markdown('---')

# ── Home vs Away ──────────────────────────────────────────────────────────────

st.markdown('### Home vs Away')

_is_home_num = pd.to_numeric(df['is_home'], errors='coerce')
_home_d = decided[_is_home_num[decided.index] == 1]
_away_d = decided[_is_home_num[decided.index] == 0]

_hwr, _hn = win_rate(_home_d)
_awr, _an = win_rate(_away_d)
_hroi = roi(_home_d)
_aroi = roi(_away_d)

_ha_cols = st.columns(2)
with _ha_cols[0]:
    _hw = int((_home_d['result'] == 'W').sum())
    _hl = int((_home_d['result'] == 'L').sum())
    st.markdown(
        f'<div style="background:#0f1f38;border:1px solid #1e3a5f;border-radius:10px;padding:14px 18px;">'
        f'<div style="font-size:13px;color:#38bdf8;font-weight:700;margin-bottom:8px;">🏠 Home Games</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color_wr(_hwr)};">'
        f'{_hwr}%</div>' if _hwr else '<div style="font-size:28px;color:#475569;">—</div>'
        f'<div style="color:#94a3b8;font-size:13px;">{_hw}–{_hl} · {_hn} plays'
        f'{(" · ROI " + str(_hroi) + "%") if _hroi else ""}</div></div>',
        unsafe_allow_html=True
    )
with _ha_cols[1]:
    _aw = int((_away_d['result'] == 'W').sum())
    _al = int((_away_d['result'] == 'L').sum())
    st.markdown(
        f'<div style="background:#0f1f38;border:1px solid #1e3a5f;border-radius:10px;padding:14px 18px;">'
        f'<div style="font-size:13px;color:#38bdf8;font-weight:700;margin-bottom:8px;">✈️ Away Games</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color_wr(_awr)};">'
        f'{_awr}%</div>' if _awr else '<div style="font-size:28px;color:#475569;">—</div>'
        f'<div style="color:#94a3b8;font-size:13px;">{_aw}–{_al} · {_an} plays'
        f'{(" · ROI " + str(_aroi) + "%") if _aroi else ""}</div></div>',
        unsafe_allow_html=True
    )

st.markdown('---')

# ── Win Rate by Pitcher Handedness ───────────────────────────────────────────

st.markdown('### Win Rate by Pitcher Handedness')

_pt_col = 'pitcher_throws'
if _pt_col not in df.columns or df[_pt_col].dropna().eq('').all():
    st.info('Pitcher handedness data is logged from today\'s games onwards — check back after the next set of results.')
else:
    _rh_d = decided[decided[_pt_col].astype(str).str.strip() == 'R']
    _lh_d = decided[decided[_pt_col].astype(str).str.strip() == 'L']
    _rwr, _rn = win_rate(_rh_d)
    _lwr, _ln = win_rate(_lh_d)
    _rroi = roi(_rh_d)
    _lroi = roi(_lh_d)

    _pt_cols = st.columns(2)
    for _col, _d, _wr, _n, _r, _label in [
        (_pt_cols[0], _rh_d, _rwr, _rn, _rroi, '🤜 vs RHP (Right-Handed)'),
        (_pt_cols[1], _lh_d, _lwr, _ln, _lroi, '🤛 vs LHP (Left-Handed)'),
    ]:
        _w = int((_d['result'] == 'W').sum()); _l = int((_d['result'] == 'L').sum())
        with _col:
            st.markdown(
                f'<div style="background:#0f1f38;border:1px solid #1e3a5f;border-radius:10px;padding:14px 18px;">'
                f'<div style="font-size:13px;color:#38bdf8;font-weight:700;margin-bottom:8px;">{_label}</div>'
                f'<div style="font-size:28px;font-weight:800;color:{color_wr(_wr)};">'
                f'{_wr}%</div>' if _wr else '<div style="font-size:28px;color:#475569;">—</div>'
                f'<div style="color:#94a3b8;font-size:13px;">{_w}–{_l} · {_n} plays'
                f'{(" · ROI " + str(_r) + "%") if _r else ""}</div></div>',
                unsafe_allow_html=True
            )

st.markdown('---')

# ── Backup ────────────────────────────────────────────────────────────────────

st.markdown('### Backup')
dl, ul = st.columns(2)
with dl:
    # Export the raw log from the database — not this page's dataframe, which
    # carries display-only benchmark results that must not be baked into a backup
    st.download_button('⬇️ Download Play Log', data=load_all().to_csv(index=False),
                       file_name='mlb_play_log.csv', mime='text/csv',
                       use_container_width=True)
with ul:
    up = st.file_uploader('⬆️ Restore Play Log', type='csv')
    if up:
        save_all(pd.read_csv(up))
        st.success('Restored!')
        st.rerun()
