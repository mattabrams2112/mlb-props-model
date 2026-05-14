"""
Umpire K-zone tendencies — affects hit rate (tight zone = fewer hits, more Ks).
Uses a static lookup of known umpire tendencies updated seasonally.
Score: +1 = hitter-friendly (wide zone, fewer Ks), -1 = pitcher-friendly (tight zone, more Ks)
"""
import os
import statsapi
import streamlit as st
from data_dir import data_path

# Static umpire tendency lookup — positive = hitter-friendly, negative = pitcher-friendly
# Based on historical K-rate deviation from league average
# Updated for 2025-2026 season (approximate)
UMPIRE_TENDENCIES = {
    'Angel Hernandez':     0.8,
    'CB Bucknor':          0.6,
    'Ron Kulpa':           0.5,
    'Hunter Wendelstedt':  0.4,
    'Jerry Meals':         0.3,
    'Dan Iassogna':        0.2,
    'Mark Carlson':        0.1,
    'Brian Gorman':        0.1,
    'Bill Miller':         0.0,
    'Mike Everitt':        0.0,
    'John Hirschbeck':     0.0,
    'Tim Timmons':        -0.1,
    'Jim Reynolds':       -0.1,
    'Laz Diaz':           -0.2,
    'Greg Gibson':        -0.2,
    'Phil Cuzzi':         -0.3,
    'Ted Barrett':        -0.3,
    'Fieldin Culbreth':   -0.4,
    'Chris Guccione':     -0.4,
    'Mike Winters':       -0.5,
    'Vic Carapazza':      -0.5,
    'Tripp Gibson':       -0.6,
    'Adam Hamari':        -0.3,
    'James Hoye':          0.2,
    'Marvin Hudson':      -0.2,
    'Brian Knight':        0.1,
    'Carlos Torres':       0.0,
}
DEFAULT_TENDENCY = 0.0


def _normalize_name(name: str) -> str:
    return name.lower().strip()


def get_umpire_tendency(ump_name: str) -> float:
    """Returns tendency score for a given umpire name."""
    for k, v in UMPIRE_TENDENCIES.items():
        if _normalize_name(k) == _normalize_name(ump_name):
            return v
    # Partial match on last name
    last = ump_name.split()[-1].lower() if ump_name else ''
    for k, v in UMPIRE_TENDENCIES.items():
        if last and last in k.lower():
            return v
    return DEFAULT_TENDENCY


@st.cache_data(show_spinner=False, ttl=3600)
def get_game_umpire(game_pk: int) -> dict:
    """Fetch home plate umpire for a game."""
    try:
        data = statsapi.get('game', {
            'gamePk': game_pk,
            'fields': 'gameData,officials'
        })
        officials = data.get('gameData', {}).get('officials', [])
        for official in officials:
            if official.get('officialType') == 'Home Plate':
                name = official.get('official', {}).get('fullName', '')
                return {
                    'umpire_name':      name,
                    'umpire_tendency':  get_umpire_tendency(name),
                }
    except Exception:
        pass
    return {'umpire_name': '', 'umpire_tendency': DEFAULT_TENDENCY}
