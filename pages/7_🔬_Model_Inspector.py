"""
Model Inspector — feature importances and CV MAE for the XGBoost projection model.
Enter any player to see which features the model actually relies on.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from data_collector import lookup_player, get_game_logs
from feature_engineering import build_features, get_feature_cols, TARGET_COL

st.set_page_config(page_title="Model Inspector | MLB Props", page_icon="🔬", layout="wide")

st.markdown("""
<style>
  h1,h2,h3{color:#38bdf8!important;}
  .stMarkdown p,label,.stCaption{color:#7dd3fc!important;}
  .stMetric label{color:#38bdf8!important;}
</style>
""", unsafe_allow_html=True)

st.markdown('## 🔬 Model Inspector')
st.caption('Feature importances and CV MAE — see exactly what the XGBoost model is using.')

player_name = st.text_input('Player name', placeholder='e.g. Freddie Freeman')

if not player_name:
    st.info('Enter a player name to inspect the model.')
    st.stop()


@st.cache_data(show_spinner=False, ttl=3600)
def run_inspection(name: str):
    player = lookup_player(name)
    df     = get_game_logs(player['id'])
    if df.empty or len(df) < 25:
        return None, player['fullName'], 0

    df_feat = build_features(df, fetch_weather=False, fast_mode=True)
    fc      = get_feature_cols(include_pitcher=False)
    dc      = df_feat.dropna(subset=fc).reset_index(drop=True)
    if len(dc) < 25:
        return None, player['fullName'], len(dc)

    X = dc.iloc[:-1][fc].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = dc.iloc[:-1][TARGET_COL]

    # Cross-validation MAE
    tscv    = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for train_idx, val_idx in tscv.split(X):
        m = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        cv_maes.append(mean_absolute_error(y.iloc[val_idx], m.predict(X.iloc[val_idx])))

    # Full model for importances
    model = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=fc).sort_values(ascending=False)

    return {
        'importances': imp,
        'cv_mae':      round(float(np.mean(cv_maes)), 3),
        'cv_maes':     [round(m, 3) for m in cv_maes],
        'n_rows':      len(X),
        'n_features':  len(fc),
        'player':      player['fullName'],
    }, player['fullName'], len(dc)


with st.spinner(f'Fetching data and training model for {player_name}...'):
    try:
        result, full_name, n_rows = run_inspection(player_name)
    except Exception as e:
        st.error(f'Could not load player: {e}')
        st.stop()

if result is None:
    st.error(f'Not enough game data for {full_name} ({n_rows} rows, need 25+).')
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric('Player',        result['player'])
c2.metric('Training Rows', result['n_rows'])
c3.metric('Features Used', result['n_features'])
c4.metric('CV MAE',        result['cv_mae'],
          help='Mean absolute error from 5-fold time-series CV. Lower = better. '
               'Typical range 0.80–1.20 for H+R+RBI.')

st.caption(f"CV MAE by fold: {' · '.join(str(m) for m in result['cv_maes'])}")

st.markdown('---')

# ── Feature importance chart ──────────────────────────────────────────────────

st.markdown('### Feature Importances — Top 30')
st.caption('Higher = model relies on this feature more. Near-zero = safe to cut.')

imp     = result['importances']
top30   = imp.head(30)
zero_ct = int((imp < 0.001).sum())

fig = go.Figure(go.Bar(
    x=top30.values[::-1],
    y=top30.index[::-1],
    orientation='h',
    marker_color=[
        '#22c55e' if v >= 0.05 else '#38bdf8' if v >= 0.01 else '#94a3b8'
        for v in top30.values[::-1]
    ],
))
fig.update_layout(
    height=700,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#7dd3fc'),
    margin=dict(l=10, r=20, t=10, b=20),
    xaxis=dict(title='Importance Score', color='#7dd3fc', gridcolor='#1e293b'),
    yaxis=dict(color='#e0f2fe', tickfont=dict(size=11)),
)
st.plotly_chart(fig, use_container_width=True)

if zero_ct > 0:
    st.caption(f'⚠️ {zero_ct} features have near-zero importance (<0.001) — candidates to cut.')

st.markdown('---')

# ── Full table ────────────────────────────────────────────────────────────────

st.markdown('### All Features')
imp_df = imp.reset_index()
imp_df.columns = ['Feature', 'Importance']
imp_df.insert(0, 'Rank', range(1, len(imp_df) + 1))
imp_df['Importance'] = imp_df['Importance'].round(4)
imp_df['Signal'] = imp_df['Importance'].apply(
    lambda v: '🟢 Strong' if v >= 0.05 else '🔵 Moderate' if v >= 0.01 else '⚪ Weak'
)
st.dataframe(imp_df, hide_index=True, use_container_width=True)
