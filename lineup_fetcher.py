"""
Fetches today's MLB lineups and probable/confirmed starting pitchers.
Team abbreviations resolved from schedule team names via lookup dict.
"""
import statsapi
from datetime import datetime

COMPLETED = {'Final', 'Game Over', 'Completed Early', 'In Progress', 'Manager Challenge'}

# Full team name → abbreviation (MLB Stats API uses full names in schedule)
TEAM_ABBR = {
    'Arizona Diamondbacks':    'ARI',
    'Atlanta Braves':          'ATL',
    'Baltimore Orioles':       'BAL',
    'Boston Red Sox':          'BOS',
    'Chicago Cubs':            'CHC',
    'Chicago White Sox':       'CWS',
    'Cincinnati Reds':         'CIN',
    'Cleveland Guardians':     'CLE',
    'Colorado Rockies':        'COL',
    'Detroit Tigers':          'DET',
    'Houston Astros':          'HOU',
    'Kansas City Royals':      'KC',
    'Los Angeles Angels':      'LAA',
    'Los Angeles Dodgers':     'LAD',
    'Miami Marlins':           'MIA',
    'Milwaukee Brewers':       'MIL',
    'Minnesota Twins':         'MIN',
    'New York Mets':           'NYM',
    'New York Yankees':        'NYY',
    'Oakland Athletics':       'OAK',
    'Athletics':               'OAK',
    'Philadelphia Phillies':   'PHI',
    'Pittsburgh Pirates':      'PIT',
    'San Diego Padres':        'SD',
    'Seattle Mariners':        'SEA',
    'San Francisco Giants':    'SF',
    'St. Louis Cardinals':     'STL',
    'Tampa Bay Rays':          'TB',
    'Texas Rangers':           'TEX',
    'Toronto Blue Jays':       'TOR',
    'Washington Nationals':    'WSH',
}


def name_to_abbr(name: str) -> str:
    return TEAM_ABBR.get(name, name[:3].upper() if name else '')


def get_todays_games(date_str: str = None) -> list:
    if date_str is None:
        date_str = datetime.now().strftime('%m/%d/%Y')
    return statsapi.schedule(date=date_str, sportId=1)


def get_game_context(game_pk: int, status: str = '',
                     home_name: str = '', away_name: str = '') -> dict:
    result = {
        'game_pk':            game_pk,
        'home_team':          name_to_abbr(home_name),
        'away_team':          name_to_abbr(away_name),
        'home_batters':       [],   # ordered list of player IDs
        'away_batters':       [],
        'home_batter_codes':  {},   # player_id -> batting order code (100=starter spot 1, 601=sub for spot 6)
        'away_batter_codes':  {},
        'home_pitcher_id':    None,
        'away_pitcher_id':    None,
        'lineups_official':   False,
        'status':             status,
    }

    game_is_live_or_final = any(s in status for s in COMPLETED)

    # Always try boxscore for any game — works for Final, In Progress, and pre-game
    if True:
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

            # Extract batting order codes from player data
            for side, key in [(home, 'home_batter_codes'), (away, 'away_batter_codes')]:
                for player_key, pdata in side.get('players', {}).items():
                    pid   = pdata.get('person', {}).get('id')
                    ocode = pdata.get('battingOrder', '')
                    if pid and ocode:
                        try:
                            result[key][int(pid)] = int(ocode)
                        except (ValueError, TypeError):
                            pass

            if not result['home_team']:
                result['home_team'] = (home.get('team', {}).get('abbreviation', '')
                                       or name_to_abbr(home.get('team', {}).get('name', '')))
            if not result['away_team']:
                result['away_team'] = (away.get('team', {}).get('abbreviation', '')
                                       or name_to_abbr(away.get('team', {}).get('name', '')))

            hp = home.get('pitchers', [])
            ap = away.get('pitchers', [])
            result['home_pitcher_id'] = hp[0] if hp else None
            result['away_pitcher_id'] = ap[0] if ap else None
            return result
        except Exception:
            pass

    # Pre-game: use schedule hydration for probable pitchers + official lineups
    try:
        sched = statsapi.get('schedule', {
            'gamePk': game_pk,
            'hydrate': 'probablePitcher,lineups',
        })
        dates = sched.get('dates', [])
        game  = dates[0].get('games', [{}])[0] if dates else {}
        teams = game.get('teams', {})

        # Try abbreviation from schedule (may or may not be present)
        h_team = teams.get('home', {}).get('team', {})
        a_team = teams.get('away', {}).get('team', {})
        result['home_team'] = (h_team.get('abbreviation', '')
                               or name_to_abbr(h_team.get('name', ''))
                               or result['home_team'])
        result['away_team'] = (a_team.get('abbreviation', '')
                               or name_to_abbr(a_team.get('name', ''))
                               or result['away_team'])

        result['home_pitcher_id'] = teams.get('home', {}).get('probablePitcher', {}).get('id')
        result['away_pitcher_id'] = teams.get('away', {}).get('probablePitcher', {}).get('id')

        lineups      = game.get('lineups', {})
        home_players = lineups.get('homePlayers', [])
        away_players = lineups.get('awayPlayers', [])
        if home_players:
            result['home_batters']    = [p['id'] for p in home_players]
            result['lineups_official'] = True
            for i, p in enumerate(home_players):
                result['home_batter_codes'][p['id']] = (i + 1) * 100
        if away_players:
            result['away_batters'] = [p['id'] for p in away_players]
            for i, p in enumerate(away_players):
                result['away_batter_codes'][p['id']] = (i + 1) * 100
    except Exception:
        pass

    # Boxscore fallback for lineups if still empty
    if not result['home_batters']:
        try:
            box  = statsapi.boxscore_data(game_pk)
            home = box.get('home', {})
            away = box.get('away', {})
            hb   = home.get('batters', [])
            ab   = away.get('batters', [])
            if hb:
                result['home_batters']    = hb
                result['lineups_official'] = True
            if ab:
                result['away_batters'] = ab
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
        ctx = get_game_context(
            game['game_id'],
            game.get('status', ''),
            home_name=game.get('home_name', ''),
            away_name=game.get('away_name', ''),
        )
        ctx['start_time']  = game.get('game_datetime', '')
        ctx['status']      = game.get('status', '')
        ctx['away_score']  = game.get('away_score', '')
        ctx['home_score']  = game.get('home_score', '')
        contexts.append(ctx)
    return contexts
