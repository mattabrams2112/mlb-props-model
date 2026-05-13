"""
Fetches today's MLB lineups and probable/confirmed starting pitchers.
Team names always come from the schedule API (reliable).
Batter/pitcher lists come from boxscore for completed/live games.
"""
import statsapi
from datetime import datetime

COMPLETED = {'Final', 'Game Over', 'Completed Early', 'In Progress', 'Manager Challenge'}


def get_todays_games(date_str: str = None) -> list:
    if date_str is None:
        date_str = datetime.now().strftime('%m/%d/%Y')
    return statsapi.schedule(date=date_str, sportId=1)


def get_game_context(game_pk: int, status: str = '') -> dict:
    result = {
        'game_pk':          game_pk,
        'home_team':        '',
        'away_team':        '',
        'home_batters':     [],
        'away_batters':     [],
        'home_pitcher_id':  None,
        'away_pitcher_id':  None,
        'lineups_official': False,
        'status':           status,
    }

    # ── Step 1: always get team names + probable pitchers from schedule ───────
    try:
        sched = statsapi.get('schedule', {
            'gamePk': game_pk,
            'hydrate': 'probablePitcher,lineups',
        })
        dates = sched.get('dates', [])
        game  = dates[0].get('games', [{}])[0] if dates else {}
        teams = game.get('teams', {})

        result['home_team'] = teams.get('home', {}).get('team', {}).get('abbreviation', '')
        result['away_team'] = teams.get('away', {}).get('team', {}).get('abbreviation', '')
        result['home_pitcher_id'] = teams.get('home', {}).get('probablePitcher', {}).get('id')
        result['away_pitcher_id'] = teams.get('away', {}).get('probablePitcher', {}).get('id')

        # Official lineups from schedule hydration (pre-game / during game)
        lineups      = game.get('lineups', {})
        home_players = lineups.get('homePlayers', [])
        away_players = lineups.get('awayPlayers', [])
        if home_players:
            result['home_batters']    = [p['id'] for p in home_players]
            result['lineups_official'] = True
        if away_players:
            result['away_batters'] = [p['id'] for p in away_players]

    except Exception:
        pass

    # ── Step 2: for completed/live games, override with actual boxscore lineup ─
    game_is_live_or_final = any(s in status for s in COMPLETED)
    if game_is_live_or_final or not result['home_batters']:
        try:
            box  = statsapi.boxscore_data(game_pk)
            home = box.get('home', {})
            away = box.get('away', {})

            hb = home.get('batters', [])
            ab = away.get('batters', [])

            if hb:
                result['home_batters']    = hb
                result['lineups_official'] = True
            if ab:
                result['away_batters'] = ab

            # Fill team names from boxscore if still missing
            if not result['home_team']:
                result['home_team'] = home.get('team', {}).get('abbreviation', '')
            if not result['away_team']:
                result['away_team'] = away.get('team', {}).get('abbreviation', '')

            # Starting pitchers (index 0 = starter)
            if not result['home_pitcher_id']:
                hp = home.get('pitchers', [])
                result['home_pitcher_id'] = hp[0] if hp else None
            if not result['away_pitcher_id']:
                ap = away.get('pitchers', [])
                result['away_pitcher_id'] = ap[0] if ap else None

        except Exception:
            pass

    return result


def get_todays_lineups(date_str: str = None) -> list:
    games    = get_todays_games(date_str)
    contexts = []
    for game in games:
        ctx = get_game_context(game['game_id'], game.get('status', ''))
        ctx['start_time'] = game.get('game_datetime', '')
        ctx['status']     = game.get('status', '')
        contexts.append(ctx)
    return contexts
