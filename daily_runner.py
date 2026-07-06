"""
Daily runner — fetches today's lineups and generates H+R+RBI predictions
for every batter. Saves results to predictions/YYYY-MM-DD.csv.

Run manually:   python daily_runner.py
Run with date:  python daily_runner.py --date 05/13/2026
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from data_collector import get_game_logs, CURRENT_YEAR
from feature_engineering import build_features, get_feature_cols, TARGET_COL
from lineup_fetcher import get_todays_lineups
from pitcher_data import get_pitcher_name, get_pitcher_season_stats
from bvp_stats import get_bvp
import statsapi

warnings.filterwarnings('ignore')
OUTPUT_DIR = 'predictions'


def train_model(df_features: pd.DataFrame, feature_cols: list):
    df_clean = df_features.dropna(subset=feature_cols).reset_index(drop=True)
    if len(df_clean) < 20:
        return None, None
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COL]
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X, y)

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, val_idx in tscv.split(X):
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        maes.append(mean_absolute_error(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
    return model, round(float(np.mean(maes)), 3)


def predict_batter(player_id: int, pitcher_id: int, is_home: bool,
                   home_team: str, feature_cols: list) -> dict | None:
    df = get_game_logs(player_id)
    if df.empty or len(df) < 25:
        return None

    df_feat = build_features(df, fetch_weather=True, override_pitcher_id=pitcher_id)
    df_feat['is_home'] = int(is_home)
    df_feat['home_team'] = home_team

    model, mae = train_model(df_feat, feature_cols)
    if model is None:
        return None

    df_clean = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
    latest = df_clean.iloc[-1:].copy()
    latest['is_home'] = int(is_home)

    pred = float(model.predict(latest[feature_cols])[0])
    pred = max(0.0, pred)

    # Recent form
    recent_10 = df.tail(10)
    recent_avg = (recent_10['h'] + recent_10['r'] + recent_10['rbi']).mean()

    p_stats = get_pitcher_season_stats(pitcher_id) if pitcher_id else {}
    bvp = get_bvp(player_id, pitcher_id) if pitcher_id else {}

    return {
        'prediction': round(pred, 2),
        'mae': mae,
        'recent_10g_avg': round(recent_avg, 2),
        'opp_era': p_stats.get('opp_era', ''),
        'opp_whip': p_stats.get('opp_whip', ''),
        'bvp_ab': bvp.get('bvp_ab', 0),
        'bvp_avg': bvp.get('bvp_avg', ''),
    }


def lookup_player_name(player_id: int) -> str:
    try:
        data = statsapi.lookup_player(player_id)
        return data[0]['fullName'] if data else str(player_id)
    except Exception:
        return str(player_id)


def run(date_str: str = None):
    from eastern_time import today_str_et
    today = today_str_et()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{today}.csv")

    print(f"\n{'='*60}")
    print(f"  MLB Daily Predictions — {today}")
    print(f"{'='*60}")

    print("Fetching lineups...")
    games = get_todays_lineups(date_str)

    if not games:
        print("No games found for today.")
        return

    feature_cols = get_feature_cols()
    rows = []

    for game in games:
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        home_pitcher_id = game.get('home_pitcher_id')
        away_pitcher_id = game.get('away_pitcher_id')
        official = game.get('lineups_official', False)
        status = game.get('status', '')

        home_pitcher_name = get_pitcher_name(home_pitcher_id) if home_pitcher_id else 'TBD'
        away_pitcher_name = get_pitcher_name(away_pitcher_id) if away_pitcher_id else 'TBD'

        print(f"\n  {away_team} @ {home_team}  [{status}]"
              + ('' if official else '  ⚠ lineups not yet official'))
        print(f"  Starters: {away_pitcher_name} (away) vs {home_pitcher_name} (home)")

        all_batters = (
            [(bid, False, away_pitcher_id) for bid in game.get('away_batters', [])] +
            [(bid, True,  home_pitcher_id) for bid in game.get('home_batters', [])]
        )

        if not all_batters:
            print("  No lineup data yet.")
            continue

        for player_id, is_home, pitcher_id in all_batters:
            name = lookup_player_name(player_id)
            result = predict_batter(player_id, pitcher_id, is_home, home_team, feature_cols)

            if result is None:
                print(f"    {name:<25}  (not enough data)")
                continue

            team = home_team if is_home else away_team
            opp_pitcher = home_pitcher_name if not is_home else away_pitcher_name

            print(f"    {name:<25}  {result['prediction']:.2f}  "
                  f"(recent avg {result['recent_10g_avg']:.2f}  MAE {result['mae']:.2f})")

            rows.append({
                'date': today,
                'player': name,
                'player_id': player_id,
                'team': team,
                'is_home': int(is_home),
                'opp_pitcher': opp_pitcher,
                'prediction_hrr': result['prediction'],
                'recent_10g_avg': result['recent_10g_avg'],
                'model_mae': result['mae'],
                'opp_era': result['opp_era'],
                'opp_whip': result['opp_whip'],
                'bvp_ab': result['bvp_ab'],
                'bvp_avg': result['bvp_avg'],
            })

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nSaved {len(rows)} predictions → {output_path}")
    else:
        print("\nNo predictions generated.")


def main():
    parser = argparse.ArgumentParser(description='MLB Daily Props Runner')
    parser.add_argument('--date', default=None, help='Date in MM/DD/YYYY format (default: today)')
    args = parser.parse_args()
    run(args.date)


if __name__ == '__main__':
    main()
