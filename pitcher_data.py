"""
Handles two things:
  1. Pitcher season stats (ERA, WHIP, K%, BB%) — cached by pitcher_id + season
  2. Starting pitcher ID per game — cached by game_pk
"""
import os
import time
import statsapi
import pandas as pd
from datetime import datetime

CURRENT_YEAR = datetime.now().year
from data_dir import data_path
PITCHER_STATS_CACHE = data_path('cache_pitcher_stats.csv')
GAME_PITCHER_CACHE  = data_path('cache_game_pitchers.csv')

LEAGUE_AVG = {
    'opp_era': 4.30,
    'opp_whip': 1.28,
    'opp_k_pct': 0.222,
    'opp_bb_pct': 0.083,
    'opp_h_per_9': 8.8,
}


def _parse_float(val, default: float) -> float:
    try:
        v = str(val).strip()
        return float(v) if v and v not in ('---', '.---', 'None', '') else default
    except (ValueError, TypeError):
        return default


# ── Pitcher season stats ──────────────────────────────────────────────────────

def _load_pitcher_cache() -> dict:
    if not os.path.exists(PITCHER_STATS_CACHE):
        return {}
    try:
        df = pd.read_csv(PITCHER_STATS_CACHE, dtype={'key': str})
        if df.empty or 'key' not in df.columns:
            return {}
        return df.set_index('key').to_dict('index')
    except Exception:
        return {}


def _save_pitcher_cache(cache: dict):
    pd.DataFrame([{'key': k, **v} for k, v in cache.items()]).to_csv(PITCHER_STATS_CACHE, index=False)


def get_pitcher_season_stats(pitcher_id: int, season: int = None) -> dict:
    if season is None:
        season = CURRENT_YEAR
    cache = _load_pitcher_cache()
    key = f"{pitcher_id}_{season}"
    # Only use cache if it has real data (not just league averages)
    if key in cache:
        cached = cache[key]
        if cached.get('opp_era', LEAGUE_AVG['opp_era']) != LEAGUE_AVG['opp_era']:
            return cached  # has real data
        # Otherwise re-fetch — cache has stale defaults

    result = LEAGUE_AVG.copy()
    try:
        import requests as _req
        resp = _req.get(
            f'https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats',
            params={'stats': 'season', 'group': 'pitching', 'season': season},
            timeout=15
        )
        resp.raise_for_status()
        stats_list = resp.json().get('stats', [])
        splits = stats_list[0].get('splits', []) if stats_list else []
        for split in splits:
            s = split.get('stat', {})
            ip = _parse_float(s.get('inningsPitched'), 0)
            bb = _parse_float(s.get('baseOnBalls'), 0)
            k = _parse_float(s.get('strikeOuts'), 0)
            h = _parse_float(s.get('hits'), 0)
            bf = _parse_float(s.get('battersFaced'), 1)

            hr  = _parse_float(s.get('homeRuns'), 0)
            hbp = _parse_float(s.get('hitByPitch'), 0)
            # FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + 3.20
            fip = round((13*hr + 3*(bb+hbp) - 2*k) / ip + 3.20, 2) if ip > 0 else 4.20
            result = {
                'opp_era':     _parse_float(s.get('era'),  LEAGUE_AVG['opp_era']),
                'opp_whip':    _parse_float(s.get('whip'), LEAGUE_AVG['opp_whip']),
                'opp_k_pct':   round(k / bf, 3) if bf > 0 else LEAGUE_AVG['opp_k_pct'],
                'opp_bb_pct':  round(bb / bf, 3) if bf > 0 else LEAGUE_AVG['opp_bb_pct'],
                'opp_h_per_9': round(h * 9 / ip, 2) if ip > 0 else LEAGUE_AVG['opp_h_per_9'],
                'opp_fip':     max(1.0, min(8.0, fip)),
                'opp_throws':  '',  # filled by get_pitcher_throws
            }
            break
    except Exception:
        pass

    cache[key] = result
    _save_pitcher_cache(cache)
    return result


def get_pitcher_rest_days(pitcher_id: int, season: int = None,
                          game_date: str = None) -> dict:
    """Returns days of rest since last appearance and a rest bonus/penalty."""
    if season is None:
        season = CURRENT_YEAR
    defaults = {'rest_days': 5, 'rest_factor': 0.0}
    cache = _load_pitcher_cache()
    today = game_date or datetime.now().strftime('%Y-%m-%d')
    key   = f'{pitcher_id}_{season}_rest_{today}'
    if key in cache:
        return cache[key]
    try:
        data   = statsapi.player_stat_data(pitcher_id, group='pitching',
                                           type='gameLog', season=season)
        splits = data.get('stats', [])
        if not splits:
            return defaults
        # Find most recent appearance before game_date
        dates = []
        for s in splits:
            gi = s.get('game', {})
            gd = gi.get('gameDate', s.get('date', ''))[:10]
            if gd and gd < today:
                dates.append(gd)
        if not dates:
            return defaults
        last_game = max(dates)
        from datetime import datetime as dt
        rest = (dt.strptime(today, '%Y-%m-%d') - dt.strptime(last_game, '%Y-%m-%d')).days
        # Short rest penalty, normal = 0, extra rest slight bonus
        if rest <= 3:
            factor = -0.8   # very short rest — significant penalty
        elif rest == 4:
            factor = -0.3   # short rest
        elif rest == 5:
            factor = 0.0    # normal
        elif rest == 6:
            factor = 0.2    # extra rest
        else:
            factor = 0.4    # well rested
        result = {'rest_days': rest, 'rest_factor': factor}
    except Exception:
        result = defaults
    cache[key] = result
    _save_pitcher_cache(cache)
    return result


def get_pitcher_throws(pitcher_id: int) -> str:
    """Returns 'L' or 'R' for pitcher handedness."""
    try:
        data = statsapi.get('person', {'personId': pitcher_id})
        return data.get('people', [{}])[0].get('pitchHand', {}).get('code', 'R')
    except Exception:
        return 'R'


def get_pitcher_last_n_starts(pitcher_id: int, n: int = 3, season: int = None) -> dict:
    """ERA and WHIP over last N starts."""
    if season is None:
        season = CURRENT_YEAR
    defaults = {'opp_last3_era': 4.30, 'opp_last3_whip': 1.28}
    cache = _load_pitcher_cache()
    key   = f"{pitcher_id}_{season}_last{n}"
    if key in cache:
        return cache[key]
    try:
        data = statsapi.player_stat_data(
            pitcher_id, group='pitching', type='gameLog', season=season)
        starts = [s for s in data.get('stats', [])
                  if int(s.get('stat', {}).get('gamesStarted', 0)) > 0][-n:]
        if not starts:
            return defaults
        total_er, total_ip, total_h, total_bb = 0, 0, 0, 0
        for s in starts:
            st = s.get('stat', {})
            total_er += _parse_float(st.get('earnedRuns'), 0)
            total_ip += _parse_float(st.get('inningsPitched'), 0)
            total_h  += _parse_float(st.get('hits'), 0)
            total_bb += _parse_float(st.get('baseOnBalls'), 0)
        result = {
            'opp_last3_era':  round((total_er * 9 / total_ip), 2) if total_ip > 0 else 4.30,
            'opp_last3_whip': round((total_h + total_bb) / total_ip, 2) if total_ip > 0 else 1.28,
        }
    except Exception:
        result = defaults
    cache[key] = result
    _save_pitcher_cache(cache)
    return result


def get_pitcher_name(pitcher_id: int) -> str:
    try:
        results = statsapi.lookup_player(pitcher_id)
        if results:
            return results[0].get('fullName', str(pitcher_id))
    except Exception:
        pass
    return str(pitcher_id)


# ── Starting pitcher per game ─────────────────────────────────────────────────

def _load_game_pitcher_cache() -> dict:
    if not os.path.exists(GAME_PITCHER_CACHE):
        return {}
    try:
        df = pd.read_csv(GAME_PITCHER_CACHE, dtype={'game_pk': str})
        if df.empty or 'game_pk' not in df.columns:
            return {}
        return df.set_index('game_pk').to_dict('index')
    except Exception:
        return {}


def _save_game_pitcher_cache(cache: dict):
    pd.DataFrame([{'game_pk': k, **v} for k, v in cache.items()]).to_csv(GAME_PITCHER_CACHE, index=False)


def get_starting_pitchers_for_games(game_pks: list, verbose: bool = True) -> dict:
    """Returns {game_pk: {'home_pitcher_id': int|None, 'away_pitcher_id': int|None}}"""
    cache = _load_game_pitcher_cache()
    missing = [str(pk) for pk in game_pks if str(pk) not in cache]

    if missing:
        if verbose:
            print(f"  Fetching starting pitchers for {len(missing)} games "
                  f"({len(game_pks) - len(missing)} cached)...")
        for i, pk in enumerate(missing):
            try:
                box = statsapi.boxscore_data(int(pk))
                home_pitchers = box.get('home', {}).get('pitchers', [])
                away_pitchers = box.get('away', {}).get('pitchers', [])
                cache[pk] = {
                    'home_pitcher_id': home_pitchers[0] if home_pitchers else None,
                    'away_pitcher_id': away_pitchers[0] if away_pitchers else None,
                }
            except Exception:
                cache[pk] = {'home_pitcher_id': None, 'away_pitcher_id': None}

            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(missing)} fetched...")
            time.sleep(0.1)

        _save_game_pitcher_cache(cache)

    return cache


# ── Pitcher game log + rolling stats ─────────────────────────────────────────

_PITCHER_LOG_CACHE: dict = {}   # (pitcher_id, season) → DataFrame, in-memory only


def get_pitcher_game_log(pitcher_id: int, season: int = None) -> pd.DataFrame:
    """Per-start game log for a pitcher. Skips relief appearances (IP < 1)."""
    if season is None:
        season = CURRENT_YEAR
    key = (pitcher_id, season)
    if key in _PITCHER_LOG_CACHE:
        return _PITCHER_LOG_CACHE[key]

    import requests as _req
    result = pd.DataFrame()
    try:
        resp = _req.get(
            f'https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats',
            params={'stats': 'gameLog', 'group': 'pitching', 'season': season},
            timeout=15,
        )
        resp.raise_for_status()
        splits = (resp.json().get('stats') or [{}])[0].get('splits', [])
        rows = []
        for s in splits:
            stat = s.get('stat', {})
            gi   = s.get('game', {})
            ip   = _parse_float(stat.get('inningsPitched'), 0)
            if ip < 1.0:    # skip short relief outings
                continue
            bb = _parse_float(stat.get('baseOnBalls'), 0)
            k  = _parse_float(stat.get('strikeOuts'), 0)
            h  = _parse_float(stat.get('hits'), 0)
            er = _parse_float(stat.get('earnedRuns'), 0)
            bf = _parse_float(stat.get('battersFaced'), max(ip * 3, 1))
            rows.append({
                'date':      gi.get('gameDate', s.get('date', ''))[:10],
                'ip':        ip,
                'era_game':  round(er * 9 / ip, 2) if ip > 0 else 0.0,
                'whip_game': round((h + bb) / ip, 2) if ip > 0 else 0.0,
                'k_pct':     round(k / bf, 3)        if bf > 0 else 0.0,
                'bb_pct':    round(bb / bf, 3)       if bf > 0 else 0.0,
                'h_per_9':   round(h * 9 / ip, 2)   if ip > 0 else 0.0,
            })
        if rows:
            result = pd.DataFrame(rows)
            result['date'] = pd.to_datetime(result['date'], errors='coerce')
            result = result.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    except Exception:
        pass

    _PITCHER_LOG_CACHE[key] = result
    return result


def get_rolling_pitcher_stats(pitcher_id: int, game_date, season: int,
                               n: int = 5) -> dict:
    """
    Blend of rolling (last n starts) and season averages.
    60% rolling + 40% season so elite pitchers aren't over-penalised
    for 1-2 bad starts while still capturing real recent form.
    Falls back to prior-season log if not enough current-season starts,
    then to LEAGUE_AVG if still nothing.
    """
    cutoff = pd.Timestamp(game_date).date() if not isinstance(game_date, type(pd.Timestamp(0).date())) else game_date

    def _recent(log, n):
        if log.empty:
            return pd.DataFrame()
        prior = log[log['date'].dt.date < cutoff]
        return prior.tail(n)

    recent = _recent(get_pitcher_game_log(pitcher_id, season), n)
    if len(recent) < 2:
        recent = _recent(get_pitcher_game_log(pitcher_id, season - 1), n)

    if recent.empty:
        return LEAGUE_AVG.copy()

    rolling = {
        'opp_era':     round(float(recent['era_game'].mean()),  2),
        'opp_whip':    round(float(recent['whip_game'].mean()), 2),
        'opp_k_pct':   round(float(recent['k_pct'].mean()),    3),
        'opp_bb_pct':  round(float(recent['bb_pct'].mean()),   3),
        'opp_h_per_9': round(float(recent['h_per_9'].mean()),  2),
        'opp_fip':     LEAGUE_AVG.get('opp_fip', 4.20),
    }

    season_stats = get_pitcher_season_stats(pitcher_id, season)

    # 60% recent form, 40% season average
    w_r, w_s = 0.60, 0.40
    blend_keys = ['opp_era', 'opp_whip', 'opp_k_pct', 'opp_bb_pct', 'opp_h_per_9', 'opp_fip']
    return {
        k: round(w_r * rolling[k] + w_s * season_stats.get(k, LEAGUE_AVG.get(k, rolling[k])), 4)
        for k in blend_keys
    }
