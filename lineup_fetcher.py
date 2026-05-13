"""
Fetches today's MLB lineups and probable/confirmed starting pitchers.
For completed or in-progress games, pulls directly from boxscore.
"""
import statsapi
from datetime import datetime

COMPLETED_STATUSES = {'Final', 'Game Over', 'Completed Early', 'In Progress', 'Manager Challenge'}


def get_todays_games(date_str: str = None) -> list:
    if date_str is None:
        date_str = datetime.now().strftime('%m/%d/%Y')
    return statsapi.schedule(date=date_str, sportId=1)


def get_game_context(game_pk: int, status: str = '') -> dict:
    result = {
        'game_pk':         game_pk,
        'home_team':       '',
        'away_team':       '',
        'home_batters':    [],
        'away_batters':    [],
        'home_pitcher_id': None,
        'away_pitcher_id': None,
        'lineups_official':False,
        'status':          status,
    }

    game_is_live_or_final = any(s in status for s in COMPLETED_STATUSES)

    # For completed/in-progress games go straight to boxscore — it has the real lineup
    if game_is_live_or_final:
        try:
            box  = statsapi.boxscore_data(game_pk)
            home = box.get('home', {}); away = box.get('away', {})
            result['home_team']    = home.get('team', {}).get('abbreviation', '')
            result['away_team']    = away.get('team', {}).get('abbreviation', '')
            result['home_batters'] = home.get('batters', [])
            result['away_batters'] = away.get('batters', [])
            hp = home.get('pitchers', []); ap = away.get('pitchers', [])
            result['home_pitcher_id'] = hp[0] if hp else None
            result['away_pitcher_id'] = ap[0] if ap else None
            result['lineups_official'] = bool(result['home_batters'])
            return result
        except Exception:
            pass

    # Pre-game: try schedule hydration for probable pitchers + official lineups
    try:
        sched = statsapi.get('schedule', {'gamePk': game_pk, 'hydrate': 'probablePitcher,lineups'})
        dates = sched.get('dates', [])
        game  = dates[0].get('games', [{}])[0] if dates else {}
        teams = game.get('teams', {})

        result['home_team'] = teams.get('home', {}).get('team', {}).get('abbreviation', '')
        result['away_team'] = teams.get('away', {}).get('team', {}).get('abbreviation', '')
        result['home_pitcher_id'] = teams.get('home', {}).get('probablePitcher', {}).get('id')
        result['away_pitcher_id'] = teams.get('away', {}).get('probablePitcher', {}).get('id')

        lineups     = game.get('lineups', {})
        home_players = lineups.get('homePlayers', [])
        away_players = lineups.get('awayPlayers', [])
        if home_players:
            result['home_batters']    = [p['id'] for p in home_players]
            result['lineups_official'] = True
        if away_players:
            result['away_batters'] = [p['id'] for p in away_players]
    except Exception:
        pass

    # Fallback to boxscore if still no batters
    if not result['home_batters']:
        try:
            box  = statsapi.boxscore_data(game_pk)
            home = box.get('home', {}); away = box.get('away', {})
            result['home_batters'] = home.get('batters', [])
            result['away_batters'] = away.get('batters', [])
            if result['home_batters']:
                result['lineups_official'] = True
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
