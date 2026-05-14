"""MLB Props Model — Streamlit Web App"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import statsapi

from data_collector import lookup_player, get_game_logs
from feature_engineering import build_features, get_feature_cols, TARGET_COL
from pitcher_data import get_pitcher_season_stats, get_pitcher_name
from bvp_stats import get_bvp
from statcast_features import get_batter_statcast, get_pitcher_statcast
from weather import get_park_factor, PARK_FACTORS
from rating import compute_rating
from lineup_fetcher import get_todays_lineups
from team_logos import get_logo, logo_img_tag

st.set_page_config(page_title="MLB Props Model", page_icon="⚾", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .player-header { display:flex; align-items:center; gap:10px; }
  .proj-number   { font-size:52px; font-weight:800; line-height:1; }
  .grade-label   { font-size:24px; font-weight:600; }

  /* Dark theme overrides */
  h1, h2, h3, h4, h5, h6 { color: #38bdf8 !important; }
  .stMarkdown p, .stMarkdown li { color: #7dd3fc; }
  label, .stSelectbox label, .stSlider label,
  .stRadio label, .stTextInput label { color: #7dd3fc !important; }
  .stMetric label { color: #38bdf8 !important; }
  .stMetric [data-testid="metric-container"] > div { color: #e0f2fe !important; }
  .stCaption { color: #7dd3fc !important; }
  div[data-testid="stSidebar"] { background-color: #0f172a; }
  div[data-testid="stSidebar"] * { color: #7dd3fc; }

  /* Lineup table */
  table { background-color: #0f172a; color: #7dd3fc; }
  th { color: #38bdf8 !important; border-bottom: 2px solid #1e40af !important; }
  tr:nth-child(even) { background-color: #1e293b; }
  tr:nth-child(odd)  { background-color: #0f172a; }

  /* Game cards */
  div[style*="border:1px solid"] {
    border-color: #1e40af !important;
    background-color: #1e293b;
    color: #7dd3fc;
  }
</style>
""", unsafe_allow_html=True)

WIND_OPTIONS = {'Calm': 0, 'Out (hitter-friendly)': 1, 'In (pitcher-friendly)': -1, 'Crosswind': 0}
PARK_LIST    = ['(Auto / Unknown)'] + sorted(PARK_FACTORS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_player_team(player_id: int) -> str:
    try:
        data = statsapi.lookup_player(player_id)
        if data:
            return data[0].get('currentTeam', {}).get('abbreviation', '')
    except Exception:
        pass
    return ''


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_game_logs(player_id: int):
    errors = []
    current_year = datetime.now().year
    seasons = [current_year - 2, current_year - 1, current_year]
    all_rows = []

    for season in seasons:
        season_err = None
        try:
            import requests as _requests
            resp = _requests.get(
                f'https://statsapi.mlb.com/api/v1/people/{player_id}/stats',
                params={'stats': 'gameLog', 'group': 'hitting', 'season': season},
                timeout=15
            )
            resp.raise_for_status()
            stats_list = resp.json().get('stats', [])
            splits = stats_list[0].get('splits', []) if stats_list else []
            errors.append(f"Season {season}: {len(splits)} splits")
            for split in splits:
                stat      = split.get('stat', {})
                game_info = split.get('game', {})
                all_rows.append({
                    'player_id': player_id,
                    'season':    season,
                    'date':      game_info.get('gameDate', split.get('date', '')),
                    'game_pk':   str(game_info.get('gamePk', '')),
                    'opponent':  split.get('opponent', {}).get('abbreviation', ''),
                    'home_team': (split.get('team', {}).get('abbreviation', '')
                                  if split.get('isHome', True)
                                  else split.get('opponent', {}).get('abbreviation', '')),
                    'is_home':   int(split.get('isHome', True)),
                    'ab':  int(stat.get('atBats', 0)),
                    'h':   int(stat.get('hits', 0)),
                    'r':   int(stat.get('runs', 0)),
                    'rbi': int(stat.get('rbi', 0)),
                    'd':   int(stat.get('doubles', 0)),
                    't':   int(stat.get('triples', 0)),
                    'hr':  int(stat.get('homeRuns', 0)),
                    'bb':  int(stat.get('baseOnBalls', 0)),
                    'k':   int(stat.get('strikeOuts', 0)),
                    'sb':  int(stat.get('stolenBases', 0)),
                })
        except Exception as season_exc:
            season_err = str(season_exc)

        if season_err:
            errors.append(f"Season {season} error: {season_err}")

    if not all_rows:
        return pd.DataFrame(), ' | '.join(errors)

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['ab'] > 0].reset_index(drop=True)
    errors.append(f"Final rows: {len(df)}")
    return df, ' | '.join(errors)


@st.cache_data(show_spinner=False, ttl=3600)
def run_model(player_id: int, pitcher_id, is_home: bool,
              park_override: str, temp: float, wind_speed: float, wind_dir_code: int):
    df, debug_msg = fetch_game_logs(player_id)
    if df.empty:
        return {'error': f'No game data. Debug: {debug_msg}'}
    if len(df) < 25:
        return {'error': f'Only {len(df)} games found (need 25). Debug: {debug_msg}'}

    df_feat = build_features(df, fetch_weather=True, override_pitcher_id=pitcher_id)
    idx = df_feat.index[-1]
    df_feat.at[idx, 'is_home']    = int(is_home)
    df_feat.at[idx, 'temp_f']     = temp
    df_feat.at[idx, 'wind_speed'] = wind_speed
    df_feat.at[idx, 'wind_dir']   = wind_dir_code
    if park_override:
        df_feat.at[idx, 'park_factor'] = get_park_factor(park_override)

    feature_cols = get_feature_cols()
    df_clean = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
    if len(df_clean) < 20:
        return {'error': f'Not enough clean rows ({len(df_clean)}) after feature engineering.'}

    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for ti, vi in tscv.split(X):
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X.iloc[ti], y.iloc[ti])
        maes.append(mean_absolute_error(y.iloc[vi], m.predict(X.iloc[vi])))

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X, y)
    if HAS_LGBM:
        lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        lgb.fit(X, y)

    latest = df_clean.iloc[-1:].copy()
    latest.at[latest.index[0], 'is_home']    = int(is_home)
    latest.at[latest.index[0], 'temp_f']     = temp
    latest.at[latest.index[0], 'wind_speed'] = wind_speed
    latest.at[latest.index[0], 'wind_dir']   = wind_dir_code
    if park_override:
        latest.at[latest.index[0], 'park_factor'] = get_park_factor(park_override)

    xgb_pred = float(xgb.predict(latest[feature_cols])[0])
    if HAS_LGBM:
        lgb_pred   = float(lgb.predict(latest[feature_cols])[0])
        projection = max(0.0, xgb_pred * 0.55 + lgb_pred * 0.45)
    else:
        projection = max(0.0, xgb_pred)
    recent_7g  = (df.tail(7)['h']  + df.tail(7)['r']  + df.tail(7)['rbi']).mean()
    recent_30g = (df.tail(30)['h'] + df.tail(30)['r'] + df.tail(30)['rbi']).mean()
    season_avg = df_clean['total_season_avg'].iloc[-1]

    # Floor — can't go below 30% of recent averages
    s_avg  = float(season_avg) if not np.isnan(season_avg) else 0
    r30    = float(recent_30g)
    floor  = max(s_avg * 0.30, r30 * 0.30)
    projection = max(projection, floor)

    # Ceiling — can't exceed 2x the 30g avg or 5.0 absolute max
    ceiling = min(5.0, max(r30 * 2.0, s_avg * 2.0, 1.5))
    projection = min(projection, ceiling)

    return {
        'projection': round(projection, 2),
        'mae':        round(float(np.mean(maes)), 3),
        'recent_7g':  round(float(recent_7g), 2),
        'recent_30g': round(float(recent_30g), 2),
        'season_avg': round(float(season_avg), 2) if not np.isnan(season_avg) else 0.0,
        'df':         df,
    }


def build_rating(result, player_id, pitcher_id, park_team, wind_speed=0, wind_dir=0):
    season = int(result['df']['season'].iloc[-1])
    b_sc  = get_batter_statcast(player_id, season)
    p_sc  = get_pitcher_statcast(pitcher_id, season) if pitcher_id else {}
    p_std = get_pitcher_season_stats(pitcher_id, season) if pitcher_id else {}
    bvp   = get_bvp(player_id, pitcher_id) if pitcher_id else {}
    return compute_rating(
        recent_7g         = result['recent_7g'],
        recent_30g        = result['recent_30g'],
        season_avg        = result['season_avg'],
        opp_era           = p_std.get('opp_era', 4.30),
        opp_whip          = p_std.get('opp_whip', 1.28),
        batter_fb_barrel  = b_sc.get('batter_fb_barrel_pct', 0.080),
        batter_bk_barrel  = b_sc.get('batter_bk_barrel_pct', 0.040),
        batter_os_barrel  = b_sc.get('batter_os_barrel_pct', 0.050),
        pitcher_fb_barrel = p_sc.get('pitcher_fb_barrel_pct', 0.080),
        pitcher_bk_barrel = p_sc.get('pitcher_bk_barrel_pct', 0.040),
        pitcher_os_barrel = p_sc.get('pitcher_os_barrel_pct', 0.050),
        batter_fb_seen    = b_sc.get('batter_fb_seen_pct', 0.55),
        batter_bk_seen    = b_sc.get('batter_bk_seen_pct', 0.25),
        batter_os_seen    = b_sc.get('batter_os_seen_pct', 0.20),
        park_factor       = get_park_factor(park_team),
        wind_speed        = wind_speed,
        wind_dir          = wind_dir,
        bvp_avg           = bvp.get('bvp_avg', 0.250),
        bvp_sample           = bvp.get('bvp_sample', 0),
        projection           = result.get('projection'),
        batter_hard_hit_pct  = b_sc.get('batter_hard_hit_pct', 0.360),
        pitcher_hard_hit_pct = p_sc.get('pitcher_hard_hit_pct', 0.360),
        batter_xba           = b_sc.get('batter_xba', 0.250),
        pitcher_xba_allowed  = p_sc.get('pitcher_xba_allowed', 0.250),
        batter_avg_ev        = b_sc.get('batter_avg_ev', 88.0),
        pitcher_avg_ev       = p_sc.get('pitcher_avg_ev', 88.0),
    ), p_std, b_sc, p_sc, bvp


def make_gauge(rating, color):
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=rating,
        number={'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar':  {'color': color, 'thickness': 0.25},
            'steps': [
                {'range': [0,  40], 'color': '#fee2e2'},
                {'range': [40, 60], 'color': '#fef9c3'},
                {'range': [60, 75], 'color': '#dcfce7'},
                {'range': [75, 100],'color': '#bbf7d0'},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(t=20, b=0, l=20, r=20),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def rating_bar_chart(components):
    labels = list(components.keys())
    scores = [v[0] for v in components.values()]
    maxes  = [v[1] for v in components.values()]
    pcts   = [s / m for s, m in zip(scores, maxes)]
    colors = ['#22c55e' if p >= 0.75 else '#eab308' if p >= 0.50 else '#ef4444' for p in pcts]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=scores, y=labels, orientation='h', marker_color=colors,
                         text=[f'{s}/{m}' for s, m in zip(scores, maxes)],
                         textposition='outside', cliponaxis=False))
    fig.add_trace(go.Bar(x=[m - s for s, m in zip(scores, maxes)], y=labels,
                         orientation='h', marker_color='#f1f5f9', showlegend=False))
    fig.update_layout(barmode='stack', height=200,
                      margin=dict(t=10, b=10, l=10, r=60),
                      xaxis=dict(range=[0, max(maxes) + 5], showticklabels=False, showgrid=False),
                      yaxis=dict(showgrid=False),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False)
    return fig


def render_lineup_table(rows):
    html = '<table style="width:100%;border-collapse:collapse;font-size:14px;">'
    html += '<tr style="background:#f1f5f9;color:#555;font-size:12px;font-weight:600;">'
    for col in ['', 'Player', 'Team', 'Venue', 'vs Pitcher', 'Rating', 'Grade', 'Proj H+R+RBI', '7g Avg', 'Opp ERA']:
        html += f'<th style="padding:8px 10px;text-align:left;border-bottom:2px solid #e2e8f0;">{col}</th>'
    html += '</tr>'
    for i, row in enumerate(rows):
        bg    = '#ffffff' if i % 2 == 0 else '#f8fafc'
        color = row['_color']
        era   = row['Opp ERA'] if isinstance(row['Opp ERA'], str) else f"{row['Opp ERA']:.2f}"
        html += f'<tr style="background:{bg};">'
        html += f'<td style="padding:8px 10px;">{logo_img_tag(row["_team"], 28)}</td>'
        html += f'<td style="padding:8px 10px;font-weight:600;">{row["Player"]}</td>'
        html += f'<td style="padding:8px 10px;">{row["Team"]}</td>'
        html += f'<td style="padding:8px 10px;">{row["Venue"]}</td>'
        html += f'<td style="padding:8px 10px;">{row["vs Pitcher"]}</td>'
        html += f'<td style="padding:8px 10px;font-size:18px;font-weight:800;color:{color};">{row["Rating"]}</td>'
        html += f'<td style="padding:8px 10px;font-weight:700;color:{color};">{row["Grade"]}</td>'
        html += f'<td style="padding:8px 10px;font-weight:700;">{row["Projected"]}</td>'
        html += f'<td style="padding:8px 10px;">{row["7g Avg"]}</td>'
        html += f'<td style="padding:8px 10px;">{era}</td>'
        html += '</tr>'
    html += '</table>'
    return html


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image('https://a.espncdn.com/i/teamlogos/mlb/500/mlb.png', width=48)
    st.title('MLB Props Model')
    st.markdown('---')
    st.markdown('**Search a Player**')
    player_input  = st.text_input('Batter', placeholder='e.g. Freddie Freeman')
    pitcher_input = st.text_input('Pitcher (optional)', placeholder='e.g. Zack Wheeler')
    venue         = st.radio('Venue', ['Home', 'Away'], horizontal=True)
    st.markdown('**Game Conditions**')
    park_sel   = st.selectbox('Ballpark', PARK_LIST)
    temp       = st.slider('Temperature (°F)', 40, 105, 72)
    wind_speed = st.slider('Wind Speed (mph)', 0, 30, 0)
    wind_dir   = st.selectbox('Wind Direction', list(WIND_OPTIONS.keys()))
    run_btn    = st.button('Search Player ⚾', type='primary', use_container_width=True)
    st.caption('Data: MLB Stats API · Baseball Savant · Statcast')


# ── Main page ─────────────────────────────────────────────────────────────────

st.markdown(f"## ⚾ MLB Props — {datetime.now().strftime('%B %d, %Y')}")

# ── Player detail card (shown when searched) ──────────────────────────────────

if run_btn and player_input:
    st.session_state['search_player'] = player_input
    st.session_state['search_pitcher'] = pitcher_input
    st.session_state['search_venue']   = venue
    st.session_state['search_park']    = park_sel
    st.session_state['search_temp']    = temp
    st.session_state['search_wind_speed'] = wind_speed
    st.session_state['search_wind_dir']   = wind_dir

if 'search_player' in st.session_state:
    player_input  = st.session_state['search_player']
    pitcher_input = st.session_state.get('search_pitcher', '')
    venue         = st.session_state.get('search_venue', 'Home')
    park_sel      = st.session_state.get('search_park', '(Auto / Unknown)')
    temp          = st.session_state.get('search_temp', 72)
    wind_speed    = st.session_state.get('search_wind_speed', 0)
    wind_dir      = st.session_state.get('search_wind_dir', 'Calm')

    with st.spinner('Loading player...'):
        try:
            player      = lookup_player(player_input)
            player_id   = player['id']
            player_name = player['fullName']
            team_abbr   = get_player_team(player_id)
        except ValueError as err:
            st.error(str(err))
            st.session_state.pop('search_player', None)
            st.stop()

    pitcher_id   = None
    pitcher_name = None
    pitcher_team = None
    if pitcher_input.strip():
        try:
            pi = lookup_player(pitcher_input)
            pitcher_id   = pi['id']
            pitcher_name = pi['fullName']
            pitcher_team = get_player_team(pitcher_id)
        except ValueError:
            st.warning(f'Pitcher "{pitcher_input}" not found — using league averages.')

    is_home      = venue == 'Home'
    park_override = '' if park_sel == '(Auto / Unknown)' else park_sel
    wind_code    = WIND_OPTIONS[wind_dir]

    with st.spinner('Running model...'):
        result = run_model(player_id, pitcher_id, is_home,
                           park_override, temp, wind_speed, wind_code)

    if 'error' in result:
        st.error(result['error'])
    else:
        r_data, p_std, b_sc, p_sc, bvp = build_rating(
            result, player_id, pitcher_id, park_override or '', wind_speed, wind_code)

        logo_url = get_logo(team_abbr)
        st.markdown(
            f'<div class="player-header">'
            f'<img src="{logo_url}" width="52" height="52">'
            f'<span style="font-size:26px;font-weight:700;">{player_name}</span>'
            f'<span style="font-size:15px;color:#888;margin-left:6px;">'
            f'{team_abbr} · {"Home" if is_home else "Away"}</span>'
            f'</div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

        cg, cp, cb = st.columns([1.2, 1, 1.8])
        with cg:
            st.markdown('**Rating**')
            st.plotly_chart(make_gauge(r_data['total'], r_data['color']),
                            use_container_width=True, config={'displayModeBar': False})
            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="grade-label" style="color:{r_data["color"]};">Grade: {r_data["grade"]}</span>'
                f'</div>', unsafe_allow_html=True)

        with cp:
            st.markdown('**Projected H + R + RBI**')
            proj_color = '#22c55e' if result['projection'] >= 3.0 else '#eab308' if result['projection'] >= 2.0 else '#ef4444'
            st.markdown(
                f'<div style="text-align:center;padding-top:20px;">'
                f'<div class="proj-number" style="color:{proj_color};">{result["projection"]}</div>'
                f'<div style="color:#888;font-size:12px;margin-top:4px;">MAE ±{result["mae"]}</div>'
                f'</div>', unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.metric('7g avg',     result['recent_7g'])
            st.metric('30g avg',    result['recent_30g'])
            st.metric('Season avg', result['season_avg'])

        with cb:
            st.markdown('**Rating Breakdown**')
            st.plotly_chart(rating_bar_chart(r_data['components']),
                            use_container_width=True, config={'displayModeBar': False})

        if pitcher_name:
            st.markdown('---')
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f'{logo_img_tag(team_abbr, 20)} **{player_name} vs pitch type**', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    'Pitch': ['FB', 'BK', 'OS'],
                    '% Seen':   [f"{b_sc.get('batter_fb_seen_pct',0):.0%}", f"{b_sc.get('batter_bk_seen_pct',0):.0%}", f"{b_sc.get('batter_os_seen_pct',0):.0%}"],
                    'Barrel%':  [f"{b_sc.get('batter_fb_barrel_pct',0):.1%}", f"{b_sc.get('batter_bk_barrel_pct',0):.1%}", f"{b_sc.get('batter_os_barrel_pct',0):.1%}"],
                }), hide_index=True, use_container_width=True)
            with sc2:
                st.markdown(f'{logo_img_tag(pitcher_team or "", 20)} **{pitcher_name} pitch arsenal**', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    'Pitch': ['FB', 'BK', 'OS'],
                    '% Thrown':       [f"{p_sc.get('pitcher_fb_thrown_pct',0):.0%}", f"{p_sc.get('pitcher_bk_thrown_pct',0):.0%}", f"{p_sc.get('pitcher_os_thrown_pct',0):.0%}"],
                    'Barrel% Allowed':[f"{p_sc.get('pitcher_fb_barrel_pct',0):.1%}", f"{p_sc.get('pitcher_bk_barrel_pct',0):.1%}", f"{p_sc.get('pitcher_os_barrel_pct',0):.1%}"],
                }), hide_index=True, use_container_width=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric('ERA',  f"{p_std.get('opp_era',0):.2f}")
            c2.metric('WHIP', f"{p_std.get('opp_whip',0):.2f}")
            c3.metric('K%',   f"{p_std.get('opp_k_pct',0):.1%}")
            c4.metric('BB%',  f"{p_std.get('opp_bb_pct',0):.1%}")
            c5.metric('H/9',  f"{p_std.get('opp_h_per_9',0):.1f}")
            if bvp.get('bvp_ab', 0) > 0:
                st.markdown(f"**Career vs {pitcher_name}:** {bvp['bvp_ab']} AB · .{int(bvp['bvp_avg']*1000):03d} AVG · {bvp['bvp_hr']} HR")

        st.markdown('---')
        st.markdown('**Last 10 Games**')
        rec = result['df'].tail(10)[['date','opponent','is_home','ab','h','r','rbi','hr','bb','k']].copy()
        rec['date']    = rec['date'].dt.strftime('%b %d')
        rec['H+R+RBI'] = rec['h'] + rec['r'] + rec['rbi']
        rec['']        = rec['is_home'].map({1: '🏠', 0: '✈'})
        rec = rec.rename(columns={'date':'Date','opponent':'Opp','ab':'AB','h':'H','r':'R','rbi':'RBI','hr':'HR','bb':'BB','k':'K'})
        st.dataframe(rec[['Date','Opp','','AB','H','R','RBI','HR','BB','K','H+R+RBI']], hide_index=True, use_container_width=True)

    st.markdown('---')

# ── Today's lineup (auto-loads) ───────────────────────────────────────────────

col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.markdown(f"### Today's Batters — Sorted by Rating")
with col_refresh:
    if st.button('🔄 Refresh Lineups', use_container_width=True,
                 help='Reloads all lineups — use this after games start to pull actual batting orders from completed games'):
        st.session_state.pop('lineup_rows', None)
        st.session_state.pop('lineup_games', None)
        st.rerun()

if 'lineup_rows' not in st.session_state:
    with st.spinner('Fetching today\'s games...'):
        games = get_todays_lineups()
        st.session_state['lineup_games'] = games

games = st.session_state.get('lineup_games', [])

if not games:
    st.warning('No MLB games found for today.')
else:
    total_batters = sum(
        len(g.get('home_batters', [])) + len(g.get('away_batters', []))
        for g in games
    )

    # Show today's schedule regardless of whether lineups are posted
    st.caption(f"{len(games)} games today · {total_batters} batters confirmed in lineups")

    # Game matchup cards
    cols = st.columns(min(len(games), 4))
    for i, game in enumerate(games):
        with cols[i % 4]:
            away = game.get('away_team', '?')
            home = game.get('home_team', '?')
            status = game.get('status', '')
            official = game.get('lineups_official', False)
            away_p = get_pitcher_name(game.get('away_pitcher_id')) if game.get('away_pitcher_id') else 'TBD'
            home_p = get_pitcher_name(game.get('home_pitcher_id')) if game.get('home_pitcher_id') else 'TBD'
            st.markdown(
                f'<div style="border:1px solid #e2e8f0;border-radius:8px;padding:10px;text-align:center;">'
                f'<div style="font-size:13px;font-weight:700;">'
                f'{logo_img_tag(away,22)}{away} @ {logo_img_tag(home,22)}{home}'
                f'</div>'
                f'<div style="font-size:11px;color:#888;margin-top:4px;">{away_p} vs {home_p}</div>'
                f'<div style="font-size:11px;margin-top:2px;">{"🏁 " + status if status in ("Final","Game Over") else "✅ Lineup official" if official else "⏳ Lineup pending"}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown('<br>', unsafe_allow_html=True)

    if total_batters == 0:
        st.info(
            '**Lineups not yet posted.** MLB teams typically release lineups 2–3 hours before first pitch. '
            'Click **Refresh Lineup** in the sidebar to check again.'
        )
    elif 'lineup_rows' not in st.session_state:
        with st.spinner(f'Building predictions for {total_batters} batters...'):
            all_rows = []
            prog = st.progress(0)
            done = 0

            for game in games:
                home_team    = game.get('home_team', '')
                away_team    = game.get('away_team', '')
                home_pitcher = game.get('home_pitcher_id')
                away_pitcher = game.get('away_pitcher_id')
                home_p_name  = get_pitcher_name(home_pitcher) if home_pitcher else 'TBD'
                away_p_name  = get_pitcher_name(away_pitcher) if away_pitcher else 'TBD'

                batters = (
                    [(bid, False, away_pitcher, away_team, home_team) for bid in game.get('away_batters', [])] +
                    [(bid, True,  home_pitcher, home_team, home_team) for bid in game.get('home_batters', [])]
                )

                for pid, is_home, opp_pid, team, park_team in batters:
                    pname, pteam = str(pid), team
                    try:
                        pd_data = statsapi.lookup_player(pid)
                        if pd_data:
                            pname = pd_data[0]['fullName']
                            pteam = pd_data[0].get('currentTeam', {}).get('abbreviation', team)
                    except Exception:
                        pass

                    res = run_model(pid, opp_pid, is_home, park_team, 72, 0, 0)
                    if res and 'error' not in res:
                        r_data, p_std, _, _, _ = build_rating(res, pid, opp_pid, park_team)
                        opp_p = away_p_name if is_home else home_p_name
                        all_rows.append({
                            '_team':      pteam,
                            '_color':     r_data['color'],
                            'Player':     pname,
                            'Team':       pteam,
                            'Venue':      '🏠 Home' if is_home else '✈ Away',
                            'vs Pitcher': opp_p,
                            'Rating':     r_data['total'],
                            'Grade':      r_data['grade'],
                            'Projected':  res['projection'],
                            '7g Avg':     res['recent_7g'],
                            'Opp ERA':    p_std.get('opp_era', '—'),
                        })

                    done += 1
                    prog.progress(done / max(total_batters, 1))

            prog.empty()
            all_rows.sort(key=lambda x: x['Rating'], reverse=True)
            st.session_state['lineup_rows'] = all_rows

            # Auto-save 60+ rated predictions to tracker
            from tracker import add_predictions
            qualified = [r for r in all_rows if r['Rating'] >= 60]
            if qualified:
                add_predictions([{
                    'player':     r['Player'],
                    'team':       r['_team'],
                    'rating':     r['Rating'],
                    'grade':      r['Grade'],
                    'projected':  r['Projected'],
                    'vs_pitcher': r['vs Pitcher'],
                } for r in qualified])

    if 'lineup_rows' in st.session_state and st.session_state['lineup_rows']:
        rows = st.session_state['lineup_rows']
        st.caption(f'{len(rows)} batters · highest to lowest rating · refresh sidebar to update')
        st.markdown(render_lineup_table(rows), unsafe_allow_html=True)
