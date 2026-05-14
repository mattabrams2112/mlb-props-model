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

# Compressed range — spots 7-9 still get meaningful points
# Max diff between #1 and #9 is only 3 pts so it doesn't dominate the rating
BATTING_ORDER_SCORES = [6, 7, 6, 6, 5, 5, 4, 4, 4]


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
    recent_20g: float    = None,   # 20-game HRR in current venue (home or away)
    recent_ba_venue: float = None, # 20-game BA in current venue
    temp_f: float        = 72.0,
    projection: float    = None,
    bp_era: float        = 4.20,
    bp_whip: float       = 1.30,
    line: float          = None,
    over_odds: int       = None,   # sportsbook over odds (American)
    home_hrr: float          = None,
    away_hrr: float          = None,
    is_home: bool            = True,
    batter_hard_hit_pct: float = 0.360,
    pitcher_hard_hit_pct: float = 0.360,
    batter_xba: float        = 0.250,
    pitcher_xba_allowed: float = 0.250,
    batter_avg_ev: float     = 88.0,
    pitcher_avg_ev: float    = 88.0,
    # New stats
    opp_fip: float           = 4.20,
    opp_last3_era: float     = 4.30,
    opp_last3_whip: float    = 1.28,
    pitcher_throws: str      = 'R',
    batter_xba_vs_rhp: float = 0.250,
    batter_xba_vs_lhp: float = 0.250,
    batter_hard_hit_vs_rhp: float = 0.360,
    batter_hard_hit_vs_lhp: float = 0.360,
    team_runs_avg: float     = 4.5,
    umpire_tendency: float   = 0.0,
    opp_def_rating: float    = 0.0,   # positive = bad defense (good for batter)
    pitcher_rest_factor: float = 0.0, # negative = short rest (good for batter)
    pitcher_gb_pct: float    = 0.430, # high GB% = bad for batter (fewer XBH)
) -> dict:
    scores = {}

    # ── Model Projection (0-30) — primary driver ─────────────────────────────
    if projection is not None:
        proj_score = min(30.0, (max(0.0, projection) / 3.5) * 30)
    else:
        proj_score = min(30.0, (season_avg / 3.0) * 30)
    scores['Projection'] = (round(proj_score, 1), 30)

    # ── Form & Hit Rate (0-15) ────────────────────────────────────────────────
    # Weighted blend: 7g (50%) + 20g venue-specific (30%) + 30g (20%)
    r20       = recent_20g if recent_20g is not None else recent_30g
    form_raw  = 0.50 * recent_7g + 0.30 * r20 + 0.20 * recent_30g
    hrr_score = min(11.0, (form_raw / 3.5) * 11)
    # Use venue-specific BA if available for more accurate hit rate
    ba_used   = recent_ba_venue if recent_ba_venue is not None else recent_ba
    ba_bonus  = max(0.0, min(4.0, (ba_used - 0.200) / (0.350 - 0.200) * 4.0))
    scores['Form & Hit Rate'] = (round(hrr_score + ba_bonus, 1), 15)

    # ── Starter Matchup (0-15) ───────────────────────────────────────────────
    # Higher ERA = worse pitcher = better for batter = higher score
    # ERA 3.0 (Ohtani) = 0pts, ERA 4.5 (avg) = 7.5pts, ERA 6.0+ = 15pts
    blended_era = (opp_era * 0.4 + opp_fip * 0.35 + opp_last3_era * 0.25)
    era_score = max(0.0, min(15.0, 15.0 * (blended_era - 3.0) / (6.0 - 3.0)))
    if bvp_sample:
        era_score = max(0.0, min(15.0, era_score + (bvp_avg - 0.250) * 12))
    scores['Starter Matchup'] = (round(era_score, 1), 15)

    # ── Platoon Advantage (0-6) ──────────────────────────────────────────────
    # Use the correct split based on pitcher handedness
    if pitcher_throws == 'L':
        plat_xba    = batter_xba_vs_lhp
        plat_hh     = batter_hard_hit_vs_lhp
    else:
        plat_xba    = batter_xba_vs_rhp
        plat_hh     = batter_hard_hit_vs_rhp
    plat_score = max(0.0, min(6.0, 3.0 + (plat_xba - 0.250) * 15 + (plat_hh - 0.360) * 10))
    scores['Platoon'] = (round(plat_score, 1), 6)

    # ── Opponent Defense (0-5, can go negative) ──────────────────────────────
    # Bad defense (more errors) = more hits reach = good for batter
    def_score = max(-3.0, min(5.0, 2.5 + opp_def_rating))
    scores['Opp Defense'] = (round(def_score, 1), 5)

    # ── Pitcher Rest & GB% (0-5) ─────────────────────────────────────────────
    # Short rest hurts pitcher (good for batter), high GB% hurts batter
    gb_penalty  = max(-3.0, min(0.0, (0.43 - pitcher_gb_pct) * 10))  # high GB = fewer XBH
    rest_bonus  = max(-3.0, min(3.0, pitcher_rest_factor * 2))        # short rest = good for batter
    pitcher_ctx = max(-3.0, min(5.0, 2.5 + rest_bonus + gb_penalty))
    scores['Pitcher Context'] = (round(pitcher_ctx, 1), 5)

    # ── Team Run Environment (0-5) ───────────────────────────────────────────
    # High-scoring team = more RBI/run opportunities
    team_score = max(0.0, min(5.0, (team_runs_avg - 3.0) / (7.0 - 3.0) * 5.0))
    scores['Team Scoring'] = (round(team_score, 1), 5)

    # ── Umpire (0-3) ─────────────────────────────────────────────────────────
    ump_score = max(0.0, min(3.0, 1.5 + umpire_tendency))
    scores['Umpire'] = (round(ump_score, 1), 3)

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
    scores['Barrel Edge'] = (round(max(0.0, min(15.0, 7.5 + barrel_edge * 150)), 1), 15)

    # ── Contact Quality — xBA + Hard Hit Rate (0-8) ──────────────────────────
    hard_hit_edge = batter_hard_hit_pct - pitcher_hard_hit_pct
    xba_edge      = batter_xba - pitcher_xba_allowed
    ev_edge       = (batter_avg_ev - pitcher_avg_ev) / 10.0
    contact_score = max(0.0, min(8.0, 4.0 + hard_hit_edge * 30 + xba_edge * 20 + ev_edge * 2))
    scores['Contact Quality'] = (round(contact_score, 1), 8)

    # ── Park & Weather (0-12) ────────────────────────────────────────────────
    park_score = max(0.0, min(7.0, (park_factor - 0.90) / (1.15 - 0.90) * 7.0))
    wind_score = max(-2.0, min(2.0, wind_dir * min(wind_speed, 20) / 20 * 2.0))
    temp_score = max(-1.0, min(1.0, (temp_f - 50) / (85 - 50) * 1.0)) if temp_f > 0 else 0
    scores['Park & Weather'] = (round(max(0.0, min(12.0, park_score + wind_score + temp_score + 1)), 1), 12)

    # ── Batting Order (0-7) — compressed range, spots 7-9 still score meaningfully
    bo_score = BATTING_ORDER_SCORES[batting_order - 1] if 1 <= batting_order <= 9 else 5
    scores['Batting Order'] = (float(bo_score), 7)

    base_total = round(min(100, max(0, sum(v[0] for v in scores.values()))))

    # ── Odds Value Edge ───────────────────────────────────────────────────────
    # Compares model's fair probability vs sportsbook's implied probability
    # Only applied when we have a line AND odds AND projection
    if line is not None and over_odds is not None and projection is not None:
        from odds_api import fair_probability, american_to_prob, edge_rating_bonus, prob_to_american
        fair_prob    = fair_probability(projection, line)
        implied_prob = american_to_prob(over_odds)
        odds_edge    = fair_prob - implied_prob
        odds_bonus   = edge_rating_bonus(odds_edge)
        fair_odds_str = str(prob_to_american(fair_prob))
        if odds_bonus != 0:
            scores['Odds Edge'] = (round(odds_bonus, 1), 12)

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

    raw_total   = sum(v[0] for v in scores.values())
    max_possible = sum(v[1] for v in scores.values())

    # Normalize to 0-100 based on actual max possible score.
    # Calibration factor of 1.25 corrects for neutral components scoring 0
    # (hot/cold, home/away, defense) which unfairly drag the percentage down.
    # Result: average player ~55, good matchup ~65-75, elite ~80-90.
    normalized = (raw_total / max_possible * 100 * 1.25) if max_possible > 0 else 0

    # Hard cap based on projection — prevents high ratings with garbage projections
    if projection is not None:
        if projection < 0.75:
            normalized = min(normalized, 35)
        elif projection < 1.25:
            normalized = min(normalized, 50)
        elif projection < 1.75:
            normalized = min(normalized, 65)

    total = round(min(100, max(0, normalized)))

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
