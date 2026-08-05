"""
Live + forecasted weather at each MLB stadium via Open-Meteo (free, no API key).
When a game_time_utc is provided, returns the hourly forecast closest to first pitch.
Domed stadiums return neutral conditions regardless.
"""
import requests
import streamlit as st
from datetime import datetime, timezone

STADIUM_COORDS = {
    'ARI': (33.4455, -112.0667),
    'ATL': (33.8908, -84.4678),
    'BAL': (39.2838, -76.6218),
    'BOS': (42.3467, -71.0972),
    'CHC': (41.9484, -87.6553),
    'CWS': (41.8299, -87.6338),
    'CIN': (39.0979, -84.5082),
    'CLE': (41.4959, -81.6854),
    'COL': (39.7559, -104.9942),
    'DET': (42.3390, -83.0485),
    'HOU': (29.7573, -95.3556),
    'KC':  (39.0517, -94.4803),
    'LAA': (33.8003, -117.8827),
    'LAD': (34.0739, -118.2400),
    'MIA': (25.7781, -80.2197),
    'MIL': (43.0280, -87.9712),
    'MIN': (44.9817, -93.2778),
    'NYM': (40.7571, -73.8458),
    'NYY': (40.8296, -73.9262),
    'OAK': (37.7516, -122.2005),
    'PHI': (39.9061, -75.1665),
    'PIT': (40.4469, -80.0057),
    'SD':  (32.7076, -117.1570),
    'SEA': (47.5914, -122.3325),
    'SF':  (37.7786, -122.3893),
    'STL': (38.6226, -90.1928),
    'TB':  (27.7683, -82.6534),
    'TEX': (32.7473, -97.0845),
    'TOR': (43.6414, -79.3894),
    'WSH': (38.8730, -77.0074),
}

# Fully enclosed year-round — weather never has an effect.
FIXED_DOME = {'TB'}

# Retractable roofs. Whether weather matters depends on the roof's state THAT DAY,
# so these can't be answered from a static set. MLB reports the state per game in
# gameData.weather.condition ("Roof Closed" / "Dome" vs a real condition), which is
# what get_roof_closed() below reads.
#
# Previously DOMED = {'TB','TOR','TEX','MIA'}: that treated TOR/TEX/MIA as always
# closed (so they never got weather even on open-roof days) and omitted
# HOU/SEA/ARI/MIL entirely (so they got live wind/temp even with the roof shut).
# Verified against 2026-08-04: HOU and ARI were both "Roof Closed" while being
# scored with live wind.
RETRACTABLE = {'TOR', 'TEX', 'MIA', 'HOU', 'SEA', 'ARI', 'MIL'}

# Back-compat for any caller still importing DOMED — parks where we default to
# neutral conditions absent a per-game reading.
DOMED = FIXED_DOME | RETRACTABLE

NEUTRAL = {
    'temp_f': 72.0, 'wind_speed': 0.0,
    'wind_dir_code': 0, 'wind_label': 'N/A',
    'condition': 'Dome', 'is_dome': True,
    'game_time_local': '',
}


def _parse_utc(time_str: str):
    """Parse ISO UTC string to aware datetime."""
    for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%MZ', '%Y-%m-%dT%H:%M'):
        try:
            dt = datetime.strptime(time_str.split('+')[0].rstrip('Z'), fmt.rstrip('Z'))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _wind_label_and_dir(degrees: float, speed: float) -> tuple:
    if speed < 3:
        return 'Calm', 0
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    label = dirs[round(degrees / 22.5) % 16]
    if label in ('S', 'SSW', 'SW', 'SSE', 'SE'):
        return f'{label} {speed:.0f} mph', 1
    elif label in ('N', 'NNE', 'NE', 'NNW', 'NW'):
        return f'{label} {speed:.0f} mph', -1
    else:
        return f'{label} {speed:.0f} mph', 0


def _weather_label(code: int) -> str:
    if code == 0:            return 'Clear ☀️'
    if code in (1, 2):      return 'Mostly Clear'
    if code == 3:            return 'Overcast'
    if code in (45, 48):    return 'Foggy 🌫'
    if code in (51, 53, 55):return 'Drizzle 🌦'
    if code in (61, 63, 65):return 'Rain 🌧'
    if code in (80, 81, 82):return 'Showers 🌦'
    if code in (95, 96, 99):return 'Thunderstorm ⛈'
    return 'Cloudy'


@st.cache_data(show_spinner=False, ttl=1800)
def get_roof_closed(game_pk) -> bool:
    """
    Roof state for a single game, read from MLB's own report.
    True = closed (or unreadable), False = open.

    Unknown deliberately resolves to True/neutral: a wrong wind reading actively
    corrupts the Park & Weather score, whereas neutral simply adds no signal.
    """
    if not game_pk:
        return True
    try:
        import statsapi
        data = statsapi.get('game', {'gamePk': int(game_pk)})
        cond = str(data.get('gameData', {}).get('weather', {}).get('condition', '')).strip()
        if not cond:
            return True
        c = cond.lower()
        return ('roof closed' in c) or ('dome' in c)
    except Exception:
        return True


@st.cache_data(show_spinner=False, ttl=1800)
def get_stadium_weather(home_team: str, game_time_utc: str = '', game_pk=None) -> dict:
    """
    Returns weather at game time if game_time_utc is provided (ISO UTC string),
    otherwise returns current conditions.

    Retractable-roof parks resolve per game via MLB's reported condition; without
    a game_pk they fall back to neutral rather than guessing.
    """
    home_team = (home_team or '').upper()

    if home_team in FIXED_DOME:
        return NEUTRAL

    if home_team in RETRACTABLE:
        if get_roof_closed(game_pk):
            return {**NEUTRAL, 'condition': 'Roof Closed'}
        # Roof open — fall through and score real conditions.

    coords = STADIUM_COORDS.get(home_team.upper())
    if not coords:
        return {**NEUTRAL, 'is_dome': False, 'condition': 'Unknown'}

    lat, lon = coords
    game_dt  = _parse_utc(game_time_utc) if game_time_utc else None

    try:
        if game_dt:
            # ── Hourly forecast at game start time ───────────────────────────
            resp = requests.get(
                'https://api.open-meteo.com/v1/forecast',
                params={
                    'latitude':         lat,
                    'longitude':        lon,
                    'hourly':           'temperature_2m,wind_speed_10m,wind_direction_10m,weather_code',
                    'temperature_unit': 'fahrenheit',
                    'wind_speed_unit':  'mph',
                    'forecast_days':    3,
                    'timezone':         'UTC',
                },
                timeout=10
            )
            resp.raise_for_status()
            h = resp.json().get('hourly', {})
            times  = h.get('time', [])
            temps  = h.get('temperature_2m', [])
            speeds = h.get('wind_speed_10m', [])
            dirs   = h.get('wind_direction_10m', [])
            codes  = h.get('weather_code', [])

            # Find index closest to game start time
            def to_dt(s):
                try:
                    return datetime.strptime(s, '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc)
                except Exception:
                    return None

            dts = [to_dt(t) for t in times]
            valid = [(i, d) for i, d in enumerate(dts) if d is not None]
            if not valid:
                raise ValueError('No valid forecast times')

            idx = min(valid, key=lambda x: abs((x[1] - game_dt).total_seconds()))[0]

            temp   = float(temps[idx])
            speed  = float(speeds[idx])
            wdeg   = float(dirs[idx])
            wcode  = int(codes[idx])

            # Format local display time
            local_hr = times[idx] if idx < len(times) else ''

        else:
            # ── Current conditions ───────────────────────────────────────────
            resp = requests.get(
                'https://api.open-meteo.com/v1/forecast',
                params={
                    'latitude':         lat,
                    'longitude':        lon,
                    'current':          'temperature_2m,wind_speed_10m,wind_direction_10m,weather_code',
                    'temperature_unit': 'fahrenheit',
                    'wind_speed_unit':  'mph',
                    'forecast_days':    1,
                },
                timeout=10
            )
            resp.raise_for_status()
            cur   = resp.json().get('current', {})
            temp  = float(cur.get('temperature_2m', 72))
            speed = float(cur.get('wind_speed_10m', 0))
            wdeg  = float(cur.get('wind_direction_10m', 0))
            wcode = int(cur.get('weather_code', 0))
            local_hr = 'Now'

        wlabel, wdir = _wind_label_and_dir(wdeg, speed)

        return {
            'temp_f':          round(temp, 1),
            'wind_speed':      round(speed, 1),
            'wind_dir_code':   wdir,
            'wind_label':      wlabel,
            'condition':       _weather_label(wcode),
            'is_dome':         False,
            'game_time_local': local_hr,
        }

    except Exception:
        return {**NEUTRAL, 'is_dome': False, 'condition': 'Unavailable'}
