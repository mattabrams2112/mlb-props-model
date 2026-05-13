"""
Live weather at each MLB stadium via Open-Meteo (free, no API key).
Domed/retractable stadiums return neutral conditions.
"""
import requests
import streamlit as st

# (latitude, longitude) for each team's home stadium
STADIUM_COORDS = {
    'ARI': (33.4455, -112.0667),  # Chase Field
    'ATL': (33.8908, -84.4678),   # Truist Park
    'BAL': (39.2838, -76.6218),   # Camden Yards
    'BOS': (42.3467, -71.0972),   # Fenway Park
    'CHC': (41.9484, -87.6553),   # Wrigley Field
    'CWS': (41.8299, -87.6338),   # Guaranteed Rate Field
    'CIN': (39.0979, -84.5082),   # Great American Ball Park
    'CLE': (41.4959, -81.6854),   # Progressive Field
    'COL': (39.7559, -104.9942),  # Coors Field
    'DET': (42.3390, -83.0485),   # Comerica Park
    'HOU': (29.7573, -95.3556),   # Minute Maid Park
    'KC':  (39.0517, -94.4803),   # Kauffman Stadium
    'LAA': (33.8003, -117.8827),  # Angel Stadium
    'LAD': (34.0739, -118.2400),  # Dodger Stadium
    'MIA': (25.7781, -80.2197),   # LoanDepot Park
    'MIL': (43.0280, -87.9712),   # American Family Field
    'MIN': (44.9817, -93.2778),   # Target Field
    'NYM': (40.7571, -73.8458),   # Citi Field
    'NYY': (40.8296, -73.9262),   # Yankee Stadium
    'OAK': (37.7516, -122.2005),  # Oakland Coliseum
    'PHI': (39.9061, -75.1665),   # Citizens Bank Park
    'PIT': (40.4469, -80.0057),   # PNC Park
    'SD':  (32.7076, -117.1570),  # Petco Park
    'SEA': (47.5914, -122.3325),  # T-Mobile Park
    'SF':  (37.7786, -122.3893),  # Oracle Park
    'STL': (38.6226, -90.1928),   # Busch Stadium
    'TB':  (27.7683, -82.6534),   # Tropicana Field
    'TEX': (32.7473, -97.0845),   # Globe Life Field
    'TOR': (43.6414, -79.3894),   # Rogers Centre
    'WSH': (38.8730, -77.0074),   # Nationals Park
}

# Stadiums that are fully enclosed (weather has no effect)
DOMED = {'TB', 'TOR', 'TEX', 'MIA'}

# Wind direction degrees → label + hitter impact
# This is approximate — true impact depends on stadium orientation
def _wind_label_and_dir(degrees: float, speed: float) -> tuple:
    """Returns (label, dir_code) where dir_code: +1=out, -1=in, 0=cross/calm"""
    if speed < 3:
        return 'Calm', 0
    dirs = ['N','NE','E','SE','S','SW','W','NW']
    label = dirs[round(degrees / 45) % 8]
    # Very rough: winds from S/SW tend to blow out at most parks (toward CF/RF)
    # winds from N/NE tend to blow in
    if label in ('S', 'SW', 'SSW'):
        return f'{label} {speed:.0f}mph', 1
    elif label in ('N', 'NE', 'NNE'):
        return f'{label} {speed:.0f}mph', -1
    else:
        return f'{label} {speed:.0f}mph', 0


@st.cache_data(show_spinner=False, ttl=1800)  # cache 30 min
def get_stadium_weather(home_team: str) -> dict:
    """Returns weather dict for the home team's stadium."""
    neutral = {
        'temp_f': 72.0,
        'wind_speed': 0.0,
        'wind_dir_code': 0,
        'wind_label': 'N/A',
        'condition': 'Dome',
        'is_dome': True,
    }

    if home_team in DOMED:
        return neutral

    coords = STADIUM_COORDS.get(home_team)
    if not coords:
        return {**neutral, 'is_dome': False, 'condition': 'Unknown'}

    lat, lon = coords
    try:
        resp = requests.get(
            'https://api.open-meteo.com/v1/forecast',
            params={
                'latitude':        lat,
                'longitude':       lon,
                'current':         'temperature_2m,wind_speed_10m,wind_direction_10m,weather_code',
                'temperature_unit':'fahrenheit',
                'wind_speed_unit': 'mph',
                'forecast_days':   1,
            },
            timeout=10
        )
        resp.raise_for_status()
        cur = resp.json().get('current', {})
        temp      = float(cur.get('temperature_2m', 72))
        wspeed    = float(cur.get('wind_speed_10m', 0))
        wdeg      = float(cur.get('wind_direction_10m', 0))
        wcode     = int(cur.get('weather_code', 0))
        wlabel, wdir = _wind_label_and_dir(wdeg, wspeed)

        condition = _weather_code(wcode)

        return {
            'temp_f':        round(temp, 1),
            'wind_speed':    round(wspeed, 1),
            'wind_dir_code': wdir,
            'wind_label':    wlabel,
            'condition':     condition,
            'is_dome':       False,
        }
    except Exception:
        return {**neutral, 'is_dome': False, 'condition': 'Unavailable'}


def _weather_code(code: int) -> str:
    if code == 0:            return 'Clear'
    if code in (1, 2, 3):   return 'Partly Cloudy'
    if code in (45, 48):    return 'Foggy'
    if code in (51,53,55):  return 'Drizzle'
    if code in (61,63,65):  return 'Rain'
    if code in (80,81,82):  return 'Showers'
    if code in (95,96,99):  return 'Thunderstorm'
    return 'Cloudy'
