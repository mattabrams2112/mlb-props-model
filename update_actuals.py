"""
Daily cron script — fetches actual H+R+RBI for all pending plays.
Run via Railway cron or manually: python update_actuals.py
"""
from full_tracker import update_actuals
from tracker import load, save, recalc_results
import requests
import statsapi
from datetime import datetime

def update_tracker_actuals():
    """Also update the main betting tracker."""
    import pandas as pd
    df    = load()
    if df.empty:
        return 0

    from eastern_time import today_str_et
    today   = today_str_et()   # ET — with UTC, today's live games count as past after 8pm ET
    updated = 0

    for i, row in df.iterrows():
        if str(row.get('actual', '')).strip() not in ('', 'nan'):
            continue
        if str(row.get('line', '')).strip() in ('', 'nan'):
            continue
        game_date = str(row.get('date', ''))[:10]
        if game_date >= today:
            continue
        try:
            players = statsapi.lookup_player(row['player'])
            if not players:
                continue
            pid  = players[0]['id']
            year = game_date[:4]
            resp = requests.get(
                f'https://statsapi.mlb.com/api/v1/people/{pid}/stats',
                params={'stats': 'gameLog', 'group': 'hitting', 'season': year},
                timeout=15
            )
            resp.raise_for_status()
            splits = (resp.json().get('stats') or [{}])[0].get('splits', [])
            for split in splits:
                gi = split.get('game', {})
                if gi.get('gameDate', split.get('date', ''))[:10] == game_date:
                    s   = split.get('stat', {})
                    hrr = int(s.get('hits',0)) + int(s.get('runs',0)) + int(s.get('rbi',0))
                    df.at[i, 'actual'] = hrr
                    updated += 1
                    break
        except Exception:
            pass

    if updated:
        df = recalc_results(df)
        save(df)
    return updated


if __name__ == '__main__':
    print(f'[{datetime.now()}] Running daily actuals update...')
    n1 = update_actuals()
    n2 = update_tracker_actuals()
    print(f'Full log updated: {n1} plays')
    print(f'Betting tracker updated: {n2} plays')
    print('Done.')
