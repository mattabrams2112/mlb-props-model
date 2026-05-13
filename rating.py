"""
Player rating engine — scores a batter 0-100 for a given matchup.

Components:
  Form & Hit Rate  (0-18) — recent HRR rolling averages + 30-day BA
  Model Projection (0-18) — XGBoost projected H+R+RBI (anchors the rating)
  Starter Matchup  (0-18) — opposing starter ERA/WHIP + BvP history
  Bullpen          (0-10) — opposing bullpen ERA/WHIP (later inning opportunity)
  Barrel Edge      (0-12) — batter barrel rate advantage over starter
  Park & Weather   (0-14) — ballpark factor + live wind/temp
  Batting Order    (0-10) — lineup position 1-5 bonus

  Line Edge Bonus  (0-10) — projection vs sportsbook line (optional, added on top)
  Max with line:   100
"""

BATTING_ORDER_SCORES = [8, 10, 8, 7, 5, 3, 2, 1, 1]


def compute_rating(
    recent_7g: float,
    recent_30g: float,
    season_avg: float,
    opp_era: float,
    opp_whip: float,
    batter_fb_barrel: float, batter_bk_barrel: float, batter_os_barrel: float,
    pitcher_fb_barrel: float, pitcher_bk_barrel: float, pitcher_os_barrel: float,
    batter_fb_seen: float,   batter_bk_seen: float,   batter_os_seen: float,
    park_factor: float,
    wind_speed: float,
    wind_dir: int,
    bvp_avg: float       = 0.250,
    bvp_sample: int      = 0,
    batting_order: int   = 0,
    recent_ba: float     = 0.250,
    temp_f: float        = 72.0,
    projection: float    = None,
    bp_era: float        = 4.20,
    bp_whip: float       = 1.30,
    line: float          = None,
    home_hrr: float      = None,   # player's avg HRR in home games
    away_hrr: float      = None,   # player's avg HRR in away games
    is_home: bool        = True,
) -> dict:
    scores = {}

    # ── Model Projection (0-30) — primary driver ─────────────────────────────
    if projection is not None:
        proj_score = min(30.0, (max(0.0, projection) / 3.5) * 30)
    else:
        proj_score = min(30.0, (season_avg / 3.0) * 30)
    scores['Projection'] = (round(proj_score, 1), 30)

    # ── Form & Hit Rate (0-15) ────────────────────────────────────────────────
    form_raw  = 0.65 * recent_7g + 0.35 * recent_30g
    hrr_score = min(11.0, (form_raw / 3.5) * 11)
    ba_bonus  = max(0.0, min(4.0, (recent_ba - 0.200) / (0.350 - 0.200) * 4.0))
    scores['Form & Hit Rate'] = (round(hrr_score + ba_bonus, 1), 15)

    # ── Starter Matchup (0-15) ───────────────────────────────────────────────
    era_score = max(0.0, min(15.0, 15.0 * (6.0 - opp_era) / (6.0 - 3.0)))
    if bvp_sample:
        era_score = max(0.0, min(15.0, era_score + (bvp_avg - 0.250) * 12))
    scores['Starter Matchup'] = (round(era_score, 1), 15)

    # ── Bullpen (0-8) ────────────────────────────────────────────────────────
    bp_era_score  = max(0.0, min(5.0, (bp_era - 3.0) / (5.5 - 3.0) * 5.0))
    bp_whip_score = max(0.0, min(3.0, (bp_whip - 1.0) / (1.8 - 1.0) * 3.0))
    scores['Bullpen'] = (round(min(8.0, bp_era_score + bp_whip_score), 1), 8)

    # ── Barrel Edge (0-10) ───────────────────────────────────────────────────
    barrel_edge = (
        batter_fb_seen * (batter_fb_barrel - pitcher_fb_barrel) +
        batter_bk_seen * (batter_bk_barrel - pitcher_bk_barrel) +
        batter_os_seen * (batter_os_barrel - pitcher_os_barrel)
    )
    scores['Barrel Edge'] = (round(max(0.0, min(10.0, 5.0 + barrel_edge * 100)), 1), 10)

    # ── Park & Weather (0-12) ────────────────────────────────────────────────
    park_score = max(0.0, min(7.0, (park_factor - 0.90) / (1.15 - 0.90) * 7.0))
    wind_score = max(-2.0, min(2.0, wind_dir * min(wind_speed, 20) / 20 * 2.0))
    temp_score = max(-1.0, min(1.0, (temp_f - 50) / (85 - 50) * 1.0)) if temp_f > 0 else 0
    scores['Park & Weather'] = (round(max(0.0, min(12.0, park_score + wind_score + temp_score + 1)), 1), 12)

    # ── Batting Order (0-10) ─────────────────────────────────────────────────
    bo_score = BATTING_ORDER_SCORES[batting_order - 1] if 1 <= batting_order <= 9 else 4
    scores['Batting Order'] = (float(bo_score), 10)

    base_total = round(min(100, max(0, sum(v[0] for v in scores.values()))))

    # ── Line Edge Bonus (0-10) — shown separately, added to base ─────────────
    line_score  = None
    line_label  = None
    if line is not None and projection is not None:
        edge = projection - line
        if edge >= 1.5:
            line_score = 10.0
            line_label = f'+{edge:.2f} STRONG OVER'
        elif edge >= 0.75:
            line_score = 7.0
            line_label = f'+{edge:.2f} over'
        elif edge >= 0.25:
            line_score = 4.0
            line_label = f'+{edge:.2f} lean over'
        elif edge >= -0.25:
            line_score = 0.0
            line_label = f'{edge:+.2f} push'
        elif edge >= -0.75:
            line_score = -4.0
            line_label = f'{edge:.2f} lean under'
        else:
            line_score = -8.0
            line_label = f'{edge:.2f} UNDER'
        scores['Line Edge'] = (round(line_score, 1), 10)

    # ── Home/Away Split (0-8, can go negative) ───────────────────────────────
    # Boost if player performs significantly better in this venue
    split_score = 0.0
    if home_hrr is not None and away_hrr is not None and (home_hrr + away_hrr) > 0:
        venue_avg   = home_hrr if is_home else away_hrr
        overall_avg = (home_hrr + away_hrr) / 2
        split_diff  = (venue_avg - overall_avg) / max(overall_avg, 0.1)
        split_score = max(-6.0, min(8.0, split_diff * 15))
    scores['Home/Away Split'] = (round(split_score, 1), 8)

    # ── Hot/Cold Streak (0-10, can go negative) ──────────────────────────────
    # Compares last 7 games vs last 30 games — recent trend matters
    if recent_7g > 0 and recent_30g > 0:
        pct_diff = (recent_7g - recent_30g) / max(recent_30g, 0.1)
        heat_score = max(-8.0, min(10.0, pct_diff * 20))
    else:
        heat_score = 0.0
    scores['Hot/Cold Streak'] = (round(heat_score, 1), 10)

    total = round(min(100, max(0, sum(v[0] for v in scores.values()))))

    grade = (
        'A+' if total >= 90 else 'A'  if total >= 85 else 'A-' if total >= 80 else
        'B+' if total >= 75 else 'B'  if total >= 70 else 'B-' if total >= 65 else
        'C+' if total >= 60 else 'C'  if total >= 55 else 'C-' if total >= 50 else
        'D+' if total >= 45 else 'D'  if total >= 40 else 'F'
    )

    color = (
        '#22c55e' if total >= 75 else
        '#eab308' if total >= 55 else
        '#ef4444'
    )

    return {
        'total':      total,
        'base_total': base_total,
        'grade':      grade,
        'color':      color,
        'components': scores,
        'line_label': line_label,
    }
