"""
MLB Props Model — Streamlit Web App
Run locally:   streamlit run app.py
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from xgboost import XGBRegressor
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Props Model",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .metric-label  { font-size: 13px; color: #888; }
  .player-header { display: flex; align-items: center; gap: 10px; }
  .rating-number { font-size: 64px; font-weight: 800; line-height: 1; }
  .grade-label   { font-size: 28px; font-weight: 600; color: #555; }
  .proj-number   { font-size: 52px; font-weight: 800; line-height: 1; }
  table.lineup td, table.lineup th { padding: 6px 10px; }
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
            data = statsapi.player_stat_data(
                player_id, group='hitting', type='gameLog', season=season
            )
            splits = data.get('stats', [])
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
        return {'error': f'No game data returned. Debug: {debug_msg}'}
    if len(df) < 25:
        return {'error': f'Only {len(df)} games found (need 25). Debug: {debug_msg}'}

    df_feat = build_features(df, fetch_weather=True, override_pitcher_id=pitcher_id)

    # Apply game-condition overrides to the most recent row
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
        return {'error': f'Not enough clean feature rows ({len(df_clean)}) after feature engineering. Need at least 20.'}

    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for ti, vi in tscv.split(X):
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X.iloc[ti], y.iloc[ti])
        maes.append(mean_absolute_error(y.iloc[vi], m.predict(X.iloc[vi])))

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X, y)

    latest = df_clean.iloc[-1:].copy()
    latest.at[latest.index[0], 'is_home']    = int(is_home)
    latest.at[latest.index[0], 'temp_f']     = temp
    latest.at[latest.index[0], 'wind_speed'] = wind_speed
    latest.at[latest.index[0], 'wind_dir']   = wind_dir_code
    if park_override:
        latest.at[latest.index[0], 'park_factor'] = get_park_factor(park_override)

    projection = max(0.0, float(model.predict(latest[feature_cols])[0]))

    recent = df.tail(10)
    recent_7  = (df.tail(7)['h']  + df.tail(7)['r']  + df.tail(7)['rbi']).mean()
    recent_30 = (df.tail(30)['h'] + df.tail(30)['r'] + df.tail(30)['rbi']).mean()
    season_avg = df_clean['total_season_avg'].iloc[-1]

    return {
        'projection': round(projection, 2),
        'mae':        round(float(np.mean(maes)), 3),
        'recent_7g':  round(recent_7, 2),
        'recent_30g': round(recent_30, 2),
        'season_avg': round(season_avg, 2) if not np.isnan(season_avg) else 0.0,
        'df':         df,
        'df_feat':    df_clean,
    }


def make_gauge(rating: int, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=rating,
        number={'font': {'size': 52, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#ccc'},
            'bar':  {'color': color, 'thickness': 0.25},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0,  40], 'color': '#fee2e2'},
                {'range': [40, 60], 'color': '#fef9c3'},
                {'range': [60, 75], 'color': '#dcfce7'},
                {'range': [75, 100],'color': '#bbf7d0'},
            ],
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=0, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def rating_breakdown_chart(components: dict) -> go.Figure:
    labels = list(components.keys())
    scores = [v[0] for v in components.values()]
    maxes  = [v[1] for v in components.values()]
    pcts   = [s / m for s, m in zip(scores, maxes)]
    colors = ['#22c55e' if p >= 0.75 else '#eab308' if p >= 0.50 else '#ef4444' for p in pcts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores, y=labels, orientation='h',
        marker_color=colors,
        text=[f'{s}/{m}' for s, m in zip(scores, maxes)],
        textposition='outside',
        cliponaxis=False,
    ))
    fig.add_trace(go.Bar(
        x=[m - s for s, m in zip(scores, maxes)], y=labels, orientation='h',
        marker_color='#f1f5f9', showlegend=False,
    ))
    fig.update_layout(
        barmode='stack', height=200,
        margin=dict(t=10, b=10, l=10, r=60),
        xaxis=dict(range=[0, max(maxes) + 5], showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image('https://a.espncdn.com/i/teamlogos/mlb/500/mlb.png', width=48)
    st.title('MLB Props Model')
    st.markdown('---')

    player_input  = st.text_input('Batter', placeholder='e.g. Freddie Freeman')
    pitcher_input = st.text_input('Pitcher (optional)', placeholder='e.g. Zack Wheeler')
    venue         = st.radio('Venue', ['Home', 'Away'], horizontal=True)

    st.markdown('**Game Conditions**')
    park_sel   = st.selectbox('Ballpark', PARK_LIST)
    temp       = st.slider('Temperature (°F)', 40, 105, 72)
    wind_speed = st.slider('Wind Speed (mph)', 0, 30, 0)
    wind_dir   = st.selectbox('Wind Direction', list(WIND_OPTIONS.keys()))

    run_btn = st.button('Run Prediction ⚾', type='primary', use_container_width=True)
    st.markdown('---')
    st.caption('Data: MLB Stats API · Baseball Savant · Statcast')


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_player, tab_lineup = st.tabs(['🔍 Player Prediction', '📋 Today\'s Lineup'])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PLAYER PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_player:
    if not player_input:
        st.info('Enter a batter name in the sidebar and click **Run Prediction**.')
        st.stop()

    if run_btn or ('last_player' in st.session_state and st.session_state.last_player == player_input):

        # Resolve player
        with st.spinner('Looking up player...'):
            try:
                player     = lookup_player(player_input)
                player_id  = player['id']
                player_name = player['fullName']
                team_abbr  = get_player_team(player_id)
            except ValueError as e:
                st.error(str(e)); st.stop()

        st.session_state.last_player = player_input

        # Resolve pitcher
        pitcher_id    = None
        pitcher_name  = None
        pitcher_team  = None
        if pitcher_input.strip():
            with st.spinner('Looking up pitcher...'):
                try:
                    pitcher_info = lookup_player(pitcher_input)
                    pitcher_id   = pitcher_info['id']
                    pitcher_name = pitcher_info['fullName']
                    pitcher_team = get_player_team(pitcher_id)
                except ValueError:
                    st.warning(f'Pitcher "{pitcher_input}" not found — using league average stats.')

        is_home       = venue == 'Home'
        park_override = '' if park_sel == '(Auto / Unknown)' else park_sel
        wind_code     = WIND_OPTIONS[wind_dir]

        with st.spinner('Fetching data & running model (first run may take a minute)...'):
            result = run_model(player_id, pitcher_id, is_home,
                               park_override, temp, wind_speed, wind_code)

        # Show debug info
        _, debug_info = fetch_game_logs(player_id)
        if debug_info:
            st.caption(f'Debug: {debug_info}')

        if result is None or 'error' in result:
            msg = result['error'] if result and 'error' in result else 'Unknown error.'
            st.error(msg)
            st.stop()

        season = int(result['df']['season'].iloc[-1])
        b_sc   = get_batter_statcast(player_id, season)
        p_sc   = get_pitcher_statcast(pitcher_id, season) if pitcher_id else {}
        p_std  = get_pitcher_season_stats(pitcher_id, season) if pitcher_id else {}
        bvp    = get_bvp(player_id, pitcher_id) if pitcher_id else {}

        # Barrel edge for rating
        pitcher_fb_barrel = p_sc.get('pitcher_fb_barrel_pct', 0.080)
        pitcher_bk_barrel = p_sc.get('pitcher_bk_barrel_pct', 0.040)
        pitcher_os_barrel = p_sc.get('pitcher_os_barrel_pct', 0.050)

        park_factor = get_park_factor(park_override) if park_override else get_park_factor(
            result['df']['home_team'].iloc[-1] if is_home else '')

        rating_data = compute_rating(
            recent_7g         = result['recent_7g'],
            recent_30g        = result['recent_30g'],
            season_avg        = result['season_avg'],
            opp_era           = p_std.get('opp_era', 4.30),
            opp_whip          = p_std.get('opp_whip', 1.28),
            batter_fb_barrel  = b_sc.get('batter_fb_barrel_pct', 0.080),
            batter_bk_barrel  = b_sc.get('batter_bk_barrel_pct', 0.040),
            batter_os_barrel  = b_sc.get('batter_os_barrel_pct', 0.050),
            pitcher_fb_barrel = pitcher_fb_barrel,
            pitcher_bk_barrel = pitcher_bk_barrel,
            pitcher_os_barrel = pitcher_os_barrel,
            batter_fb_seen    = b_sc.get('batter_fb_seen_pct', 0.55),
            batter_bk_seen    = b_sc.get('batter_bk_seen_pct', 0.25),
            batter_os_seen    = b_sc.get('batter_os_seen_pct', 0.20),
            park_factor       = park_factor,
            wind_speed        = wind_speed,
            wind_dir          = wind_code,
            bvp_avg           = bvp.get('bvp_avg', 0.250),
            bvp_sample        = bvp.get('bvp_sample', 0),
        )

        # ── Player header ─────────────────────────────────────────────────────
        logo_url = get_logo(team_abbr)
        st.markdown(
            f'<div class="player-header">'
            f'<img src="{logo_url}" width="52" height="52">'
            f'<span style="font-size:28px; font-weight:700;">{player_name}</span>'
            f'<span style="font-size:16px; color:#888; margin-left:4px;">'
            f'{team_abbr} · {"Home" if is_home else "Away"}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('<br>', unsafe_allow_html=True)

        # ── Top metrics row ───────────────────────────────────────────────────
        col_gauge, col_proj, col_breakdown = st.columns([1.2, 1, 1.8])

        with col_gauge:
            st.markdown('**Player Rating**')
            st.plotly_chart(make_gauge(rating_data['total'], rating_data['color']),
                            use_container_width=True, config={'displayModeBar': False})
            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="grade-label" style="color:{rating_data["color"]};">'
                f'Grade: {rating_data["grade"]}</span></div>',
                unsafe_allow_html=True
            )

        with col_proj:
            st.markdown('**Projected H + R + RBI**')
            proj_color = (
                '#22c55e' if result['projection'] >= 3.0 else
                '#eab308' if result['projection'] >= 2.0 else '#ef4444'
            )
            st.markdown(
                f'<div style="text-align:center; padding-top:30px;">'
                f'<div class="proj-number" style="color:{proj_color};">'
                f'{result["projection"]}</div>'
                f'<div style="color:#888; font-size:13px; margin-top:6px;">'
                f'Model MAE: ±{result["mae"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown('<br>', unsafe_allow_html=True)
            st.metric('7-game avg',  result['recent_7g'])
            st.metric('30-game avg', result['recent_30g'])
            st.metric('Season avg',  result['season_avg'])

        with col_breakdown:
            st.markdown('**Rating Breakdown**')
            st.plotly_chart(rating_breakdown_chart(rating_data['components']),
                            use_container_width=True, config={'displayModeBar': False})

        st.markdown('---')

        # ── Batter vs Pitcher Statcast ────────────────────────────────────────
        st.markdown('### Statcast Breakdown')
        sc_col1, sc_col2 = st.columns(2)

        with sc_col1:
            batter_logo = f'<img src="{logo_url}" width="24" height="24" style="vertical-align:middle; margin-right:6px;">'
            st.markdown(f'{batter_logo} **{player_name} — vs Each Pitch Type**', unsafe_allow_html=True)
            batter_sc_df = pd.DataFrame({
                'Pitch Type': ['Fastball (FB)', 'Breaking (BK)', 'Offspeed (OS)'],
                '% Seen': [
                    f"{b_sc.get('batter_fb_seen_pct', 0):.0%}",
                    f"{b_sc.get('batter_bk_seen_pct', 0):.0%}",
                    f"{b_sc.get('batter_os_seen_pct', 0):.0%}",
                ],
                'Barrel %': [
                    f"{b_sc.get('batter_fb_barrel_pct', 0):.1%}",
                    f"{b_sc.get('batter_bk_barrel_pct', 0):.1%}",
                    f"{b_sc.get('batter_os_barrel_pct', 0):.1%}",
                ],
            })
            st.dataframe(batter_sc_df, hide_index=True, use_container_width=True)

        with sc_col2:
            if pitcher_name and p_sc:
                pitcher_logo = get_logo(pitcher_team or '')
                p_logo_tag = f'<img src="{pitcher_logo}" width="24" height="24" style="vertical-align:middle; margin-right:6px;">'
                st.markdown(f'{p_logo_tag} **{pitcher_name} — Pitch Arsenal**', unsafe_allow_html=True)
                pitcher_sc_df = pd.DataFrame({
                    'Pitch Type': ['Fastball (FB)', 'Breaking (BK)', 'Offspeed (OS)'],
                    '% Thrown': [
                        f"{p_sc.get('pitcher_fb_thrown_pct', 0):.0%}",
                        f"{p_sc.get('pitcher_bk_thrown_pct', 0):.0%}",
                        f"{p_sc.get('pitcher_os_thrown_pct', 0):.0%}",
                    ],
                    'Barrel % Allowed': [
                        f"{p_sc.get('pitcher_fb_barrel_pct', 0):.1%}",
                        f"{p_sc.get('pitcher_bk_barrel_pct', 0):.1%}",
                        f"{p_sc.get('pitcher_os_barrel_pct', 0):.1%}",
                    ],
                })
                st.dataframe(pitcher_sc_df, hide_index=True, use_container_width=True)
            else:
                st.info('Enter a pitcher name to see their pitch arsenal.')

        # ── Pitcher season stats + BvP ────────────────────────────────────────
        if pitcher_name and p_std:
            st.markdown('---')
            st.markdown('### Pitcher Stats & Matchup History')
            pc1, pc2, pc3, pc4, pc5 = st.columns(5)
            pc1.metric('ERA',  f"{p_std.get('opp_era', 0):.2f}")
            pc2.metric('WHIP', f"{p_std.get('opp_whip', 0):.2f}")
            pc3.metric('K%',   f"{p_std.get('opp_k_pct', 0):.1%}")
            pc4.metric('BB%',  f"{p_std.get('opp_bb_pct', 0):.1%}")
            pc5.metric('H/9',  f"{p_std.get('opp_h_per_9', 0):.1f}")

            if bvp.get('bvp_ab', 0) > 0:
                st.markdown(
                    f"**Career vs {pitcher_name}:** "
                    f"{bvp['bvp_ab']} AB · "
                    f".{int(bvp['bvp_avg'] * 1000):03d} AVG · "
                    f"{bvp['bvp_hr']} HR"
                    + (' *(small sample)*' if not bvp.get('bvp_sample') else '')
                )
            else:
                st.caption(f'No prior plate appearances vs {pitcher_name}.')

        # ── Recent form table ─────────────────────────────────────────────────
        st.markdown('---')
        st.markdown('### Recent 10 Games')
        recent_df = result['df'].tail(10)[['date', 'opponent', 'is_home', 'ab', 'h', 'r', 'rbi', 'hr', 'bb', 'k']].copy()
        recent_df['date']    = recent_df['date'].dt.strftime('%b %d')
        recent_df['H+R+RBI'] = recent_df['h'] + recent_df['r'] + recent_df['rbi']
        recent_df['home']    = recent_df['is_home'].map({1: '🏠', 0: '✈'})
        recent_df = recent_df.rename(columns={'date': 'Date', 'opponent': 'Opp', 'home': '', 'ab': 'AB',
                                               'h': 'H', 'r': 'R', 'rbi': 'RBI', 'hr': 'HR', 'bb': 'BB', 'k': 'K'})
        st.dataframe(recent_df[['Date', 'Opp', '', 'AB', 'H', 'R', 'RBI', 'HR', 'BB', 'K', 'H+R+RBI']],
                     hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TODAY'S LINEUP
# ══════════════════════════════════════════════════════════════════════════════

with tab_lineup:
    st.markdown(f"### Today's Lineups — {datetime.now().strftime('%B %d, %Y')}")
    st.caption('Predictions run for all confirmed batters. Sorted by rating (highest first).')

    load_btn = st.button('Load / Refresh Today\'s Lineups', type='primary')

    if load_btn or 'lineup_results' in st.session_state:

        if load_btn:
            with st.spinner('Fetching lineups from MLB API...'):
                games = get_todays_lineups()
            st.session_state['lineup_games'] = games

        games = st.session_state.get('lineup_games', [])

        if not games:
            st.warning('No games found today.')
        else:
            all_rows = []
            progress = st.progress(0)
            total_batters = sum(
                len(g.get('home_batters', [])) + len(g.get('away_batters', []))
                for g in games
            )
            done = 0

            for game in games:
                home_team      = game.get('home_team', '')
                away_team      = game.get('away_team', '')
                home_pitcher   = game.get('home_pitcher_id')
                away_pitcher   = game.get('away_pitcher_id')
                official       = game.get('lineups_official', False)
                home_p_name    = get_pitcher_name(home_pitcher) if home_pitcher else 'TBD'
                away_p_name    = get_pitcher_name(away_pitcher) if away_pitcher else 'TBD'

                batters = (
                    [(bid, False, away_pitcher, away_team, home_team) for bid in game.get('away_batters', [])] +
                    [(bid, True,  home_pitcher, home_team, home_team) for bid in game.get('home_batters', [])]
                )

                for player_id, is_home, opp_pitcher_id, team, park_team in batters:
                    try:
                        player_data = statsapi.lookup_player(player_id)
                        pname = player_data[0]['fullName'] if player_data else str(player_id)
                        pteam = player_data[0].get('currentTeam', {}).get('abbreviation', team)
                    except Exception:
                        pname = str(player_id)
                        pteam = team

                    result = run_model(player_id, opp_pitcher_id, is_home,
                                       park_team, 72, 0, 0)

                    if result:
                        season = int(result['df']['season'].iloc[-1])
                        b_sc   = get_batter_statcast(player_id, season)
                        p_sc   = get_pitcher_statcast(opp_pitcher_id, season) if opp_pitcher_id else {}
                        p_std  = get_pitcher_season_stats(opp_pitcher_id, season) if opp_pitcher_id else {}
                        bvp    = get_bvp(player_id, opp_pitcher_id) if opp_pitcher_id else {}

                        r = compute_rating(
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
                            wind_speed        = 0, wind_dir=0,
                            bvp_avg           = bvp.get('bvp_avg', 0.250),
                            bvp_sample        = bvp.get('bvp_sample', 0),
                        )

                        opp_p_name = away_p_name if is_home else home_p_name
                        all_rows.append({
                            '_team':       pteam,
                            '_color':      r['color'],
                            'Player':      pname,
                            'Team':        pteam,
                            'Venue':       '🏠 Home' if is_home else '✈ Away',
                            'vs Pitcher':  opp_p_name,
                            'Rating':      r['total'],
                            'Grade':       r['grade'],
                            'Projected':   result['projection'],
                            '7g Avg':      result['recent_7g'],
                            'Opp ERA':     p_std.get('opp_era', '—'),
                        })

                    done += 1
                    progress.progress(done / max(total_batters, 1))

            progress.empty()

            if all_rows:
                all_rows.sort(key=lambda x: x['Rating'], reverse=True)
                st.session_state['lineup_results'] = all_rows

                # Render HTML table with logos
                html = '<table style="width:100%; border-collapse:collapse;">'
                html += '<tr style="background:#f8fafc; font-size:13px; color:#555;">'
                for col in ['', 'Player', 'Team', 'Venue', 'vs Pitcher', 'Rating', 'Grade', 'Projected H+R+RBI', '7g Avg', 'Opp ERA']:
                    html += f'<th style="padding:8px 10px; text-align:left; border-bottom:2px solid #e2e8f0;">{col}</th>'
                html += '</tr>'

                for i, row in enumerate(all_rows):
                    bg = '#ffffff' if i % 2 == 0 else '#f8fafc'
                    color = row['_color']
                    logo  = logo_img_tag(row['_team'], size=28)
                    html += f'<tr style="background:{bg}; font-size:14px;">'
                    html += f'<td style="padding:8px 10px;">{logo}</td>'
                    html += f'<td style="padding:8px 10px; font-weight:600;">{row["Player"]}</td>'
                    html += f'<td style="padding:8px 10px;">{row["Team"]}</td>'
                    html += f'<td style="padding:8px 10px;">{row["Venue"]}</td>'
                    html += f'<td style="padding:8px 10px;">{row["vs Pitcher"]}</td>'
                    html += f'<td style="padding:8px 10px;"><span style="font-size:18px; font-weight:800; color:{color};">{row["Rating"]}</span></td>'
                    html += f'<td style="padding:8px 10px;"><span style="font-weight:700; color:{color};">{row["Grade"]}</span></td>'
                    html += f'<td style="padding:8px 10px; font-weight:700;">{row["Projected"]}</td>'
                    html += f'<td style="padding:8px 10px;">{row["7g Avg"]}</td>'
                    era_display = row["Opp ERA"] if isinstance(row["Opp ERA"], str) else f'{row["Opp ERA"]:.2f}'
                    html += f'<td style="padding:8px 10px;">{era_display}</td>'
                    html += '</tr>'
                html += '</table>'

                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info('No predictions could be generated — lineups may not be posted yet.')
    else:
        st.info('Click **Load / Refresh Today\'s Lineups** to fetch predictions for all of today\'s batters.')
