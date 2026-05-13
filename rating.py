"""
Player rating engine — scores a batter 0-100 for a given matchup.

Components:
  Form & Hit Rate  (0-20) — recent H+R+RBI rolling averages + 30-day BA
  Model Projection (0-20) — XGBoost projected H+R+RBI (keeps rating grounded)
  Matchup          (0-22) — opposing pitcher quality + BvP history
  Barrel Edge      (0-13) — batter barrel rate advantage over pitcher
  Park & Weather   (0-15) — ballpark factor + live wind/temp
  Batting Order    (0-10) — lineup position 1-5 bonus
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
    bvp_avg: float      = 0.250,
    bvp_sample: int     = 0,
    batting_order: int  = 0,
    recent_ba: float    = 0.250,
    temp_f: float       = 72.0,
    projection: float   = None,   # XGBoost model output — anchors the rating
) -> dict:
    scores = {}

    # ── Form & Hit Rate (0-20) ────────────────────────────────────────────────
    form_raw  = 0.65 * recent_7g + 0.35 * recent_30g
    hrr_score = min(16.0, (form_raw / 3.5) * 16)
    ba_bonus  = max(0.0, min(4.0, (recent_ba - 0.200) / (0.350 - 0.200) * 4.0))
    scores['Form & Hit Rate'] = (round(hrr_score + ba_bonus, 1), 20)

    # ── Model Projection (0-20) ───────────────────────────────────────────────
    # This anchors the rating to the actual model output.
    # 0.0 proj = 0 pts, 2.0 = 11 pts, 3.5+ = 20 pts
    if projection is not None:
        proj_score = min(20.0, (max(0.0, projection) / 3.5) * 20)
    else:
        # Fall back to season avg if no projection available
        proj_score = min(20.0, (season_avg / 3.0) * 20)
    scores['Projection'] = (round(proj_score, 1), 20)

    # ── Matchup (0-22) ───────────────────────────────────────────────────────
    era_score = max(0.0, min(22.0, 22.0 * (6.0 - opp_era) / (6.0 - 3.0)))
    if bvp_sample:
        era_score = max(0.0, min(22.0, era_score + (bvp_avg - 0.250) * 18))
    scores['Matchup'] = (round(era_score, 1), 22)

    # ── Barrel Edge (0-13) ───────────────────────────────────────────────────
    barrel_edge = (
        batter_fb_seen * (batter_fb_barrel - pitcher_fb_barrel) +
        batter_bk_seen * (batter_bk_barrel - pitcher_bk_barrel) +
        batter_os_seen * (batter_os_barrel - pitcher_os_barrel)
    )
    scores['Barrel Edge'] = (round(max(0.0, min(13.0, 6.5 + barrel_edge * 130)), 1), 13)

    # ── Park & Weather (0-15) ────────────────────────────────────────────────
    park_score = max(0.0, min(8.0, (park_factor - 0.90) / (1.15 - 0.90) * 8.0))
    wind_score = max(-3.0, min(3.0, wind_dir * min(wind_speed, 20) / 20 * 3.0))
    temp_score = max(-2.0, min(2.0, (temp_f - 50) / (85 - 50) * 2.0)) if temp_f > 0 else 0
    scores['Park & Weather'] = (round(max(0.0, min(15.0, park_score + wind_score + temp_score + 2)), 1), 15)

    # ── Batting Order (0-10) ─────────────────────────────────────────────────
    bo_score = BATTING_ORDER_SCORES[batting_order - 1] if 1 <= batting_order <= 9 else 4
    scores['Batting Order'] = (float(bo_score), 10)

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

    return {'total': total, 'grade': grade, 'color': color, 'components': scores}
