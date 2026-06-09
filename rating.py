"""
Player rating engine — scores a batter 0-100 for a given matchup.

Components:
  Model Projection  (0-25) — XGBoost projected H+R+RBI (primary driver)
  Starter Matchup   (0-22) — opposing starter ERA/FIP/WHIP + BvP history
  Form & Hit Rate   (0-15) — recent HRR rolling averages + venue BA
  Platoon           (-4–10) — xBA + hard hit rate vs pitcher handedness
  Hot/Cold Streak   (-8–10) — last 7g vs last 30g trend
  Home/Away Split   (-6–8)  — player's venue-specific HRR history
  Batted Ball Edge  (0-12) — barrel rate + xBA + hard hit + whiff matchup
  Bullpen           (0-8)  — opposing bullpen ERA/WHIP
  Park & Weather    (0-9)  — ballpark factor + wind/temp
  Batting Order     (0-7)  — lineup position (wider spread: 2-7pts)
  Team Scoring      (0-5)  — team's avg runs scored
  Opp Defense       (-3–5) — opposing team's defense rating
  Pitcher Context   (-3–5) — pitcher rest + ground ball %
  Batter Rest       (-2–1) — days since last game
  Line Edge         — projection vs sportsbook line (label only, not scored)
"""

# Wider spread so batting order actually matters: leadoff/2-hole get max,
# bottom of order gets meaningfully less
BATTING_ORDER_SCORES = [7, 7, 6, 6, 5, 4, 3, 2, 2]


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
    batter_rest_days: int    = 1,     # days since last game (0=back-to-back, 1=normal)
    batter_obp: float        = 0.320, # rolling OBP (H+BB)/(AB+BB)
    batter_sb_rate: float    = 0.05,  # average stolen bases per game (30g)
    pitcher_last_pitch_count: int = 0, # pitches thrown in last start (0 = unknown)
    park_ba: float           = 0.250,  # batter career BA at this park
    park_slg: float          = 0.400,  # batter career SLG at this park
    park_ab: int             = 0,      # career AB at this park (0 = no history)
    # Contact discipline — pitcher side
    opp_k_pct: float           = 0.222,
    opp_bb_pct: float          = 0.083,
    opp_babip: float           = 0.300,
    opp_whiff_pct: float       = 0.245,
    opp_k_pct_vs_lhb: float    = None,
    opp_k_pct_vs_rhb: float    = None,
    opp_babip_vs_lhb: float    = None,
    opp_babip_vs_rhb: float    = None,
    # Per-pitch-group whiff% for weighted matchup calculation
    pitcher_fb_thrown: float   = 0.55,
    pitcher_bk_thrown: float   = 0.25,
    pitcher_os_thrown: float   = 0.20,
    batter_whiff_pct_fb: float = 0.200,
    batter_whiff_pct_bk: float = 0.330,
    batter_whiff_pct_os: float = 0.310,
    pitcher_whiff_pct_fb: float = 0.200,
    pitcher_whiff_pct_bk: float = 0.330,
    pitcher_whiff_pct_os: float = 0.310,
    # Contact discipline — batter side
    batter_k_pct: float        = 0.222,
    batter_bb_pct: float       = 0.083,
    batter_babip: float        = 0.300,
    batter_whiff_pct: float    = 0.245,
    batter_k_pct_vs_rhp: float = None,
    batter_k_pct_vs_lhp: float = None,
    batter_babip_vs_rhp: float = None,
    batter_babip_vs_lhp: float = None,
) -> dict:
    scores = {}

    # ── Model Projection (0-25) — primary driver ─────────────────────────────
    if projection is not None:
        proj_score = min(25.0, (max(0.0, projection) / 3.5) * 25)
    else:
        proj_score = min(25.0, (season_avg / 3.0) * 25)
    scores['Projection'] = (round(proj_score, 1), 25)

    # ── Form & Hit Rate (0-15) ────────────────────────────────────────────────
    # Weighted blend: 7g (20%) + 20g venue-specific (40%) + 30g (40%)
    r20       = recent_20g if recent_20g is not None else recent_30g
    form_raw  = 0.20 * recent_7g + 0.40 * r20 + 0.40 * recent_30g
    hrr_score = min(11.0, (form_raw / 3.5) * 11)
    # OBP is more predictive than BA for runs (captures walk-drawing); SB directly creates runs
    _obp      = batter_obp if batter_obp > 0 else (recent_ba * 1.05 + 0.03)
    _obp_bonus = max(0.0, min(3.5, (_obp - 0.280) / (0.420 - 0.280) * 3.5))
    _sb_bonus  = min(0.5, batter_sb_rate * 3.0)
    scores['Form & Hit Rate'] = (round(min(15.0, hrr_score + _obp_bonus + _sb_bonus), 1), 15)

    # ── Starter Matchup (0-20) ───────────────────────────────────────────────
    # Higher ERA = worse pitcher = better for batter = higher score
    # ERA 2.5 (elite) = 0pts, ERA 4.5 (avg) = 10pts, ERA 6.5+ = 20pts
    # Season ERA (40%) + FIP (35%) carry most weight, last 3 starts (25%) small recency bump
    # If last3 ERA is the league default (4.30), fall back to season ERA
    _last3_era  = opp_last3_era if abs(opp_last3_era - 4.30) > 0.05 else opp_era
    blended_era = (opp_era * 0.40 + opp_fip * 0.35 + _last3_era * 0.25)
    era_score = max(0.0, min(22.0, 22.0 * (blended_era - 2.5) / (6.5 - 2.5)))
    if bvp_sample:
        era_score = max(0.0, min(22.0, era_score + (bvp_avg - 0.250) * 15))
    # K% modifier: elite K% pitcher reduces batter opportunity (-3 to +2 pts)
    _eff_k_pct   = (opp_k_pct_vs_lhb if pitcher_throws == 'L' and opp_k_pct_vs_lhb is not None
                    else opp_k_pct_vs_rhb if pitcher_throws == 'R' and opp_k_pct_vs_rhb is not None
                    else opp_k_pct)
    _eff_babip   = (opp_babip_vs_lhb if pitcher_throws == 'L' and opp_babip_vs_lhb is not None
                    else opp_babip_vs_rhb if pitcher_throws == 'R' and opp_babip_vs_rhb is not None
                    else opp_babip)
    k_adj    = max(-3.0, min(2.0, (0.222 - _eff_k_pct) * 20))
    babip_adj = max(-1.5, min(1.5, (_eff_babip - 0.300) * 5))
    era_score = max(0.0, min(22.0, era_score + k_adj + babip_adj))
    scores['Starter Matchup'] = (round(era_score, 1), 22)

    # ── Platoon Advantage (0-6) ──────────────────────────────────────────────
    # Use the correct split based on pitcher handedness
    if pitcher_throws == 'L':
        plat_xba    = batter_xba_vs_lhp
        plat_hh     = batter_hard_hit_vs_lhp
    else:
        plat_xba    = batter_xba_vs_rhp
        plat_hh     = batter_hard_hit_vs_rhp
    # Batter K% and BABIP vs this pitcher's handedness
    _bat_k    = (batter_k_pct_vs_rhp if pitcher_throws == 'R' and batter_k_pct_vs_rhp is not None
                 else batter_k_pct_vs_lhp if pitcher_throws == 'L' and batter_k_pct_vs_lhp is not None
                 else batter_k_pct)
    _bat_babip = (batter_babip_vs_rhp if pitcher_throws == 'R' and batter_babip_vs_rhp is not None
                  else batter_babip_vs_lhp if pitcher_throws == 'L' and batter_babip_vs_lhp is not None
                  else batter_babip)
    k_plat     = max(-1.5, min(1.5, (0.222 - _bat_k)    * 7))
    babip_plat = max(-1.0, min(1.0, (_bat_babip - 0.300) * 4))

    # Pitch-mix-weighted barrel edge: batter's barrel rate vs pitch type ×
    # pitcher's actual usage of that pitch type — captures e.g. "lefty who
    # throws 70% sliders against a batter who barrels sliders at 12%"
    _total_thrown = pitcher_fb_thrown + pitcher_bk_thrown + pitcher_os_thrown
    if _total_thrown > 0:
        _fb_w = pitcher_fb_thrown / _total_thrown
        _bk_w = pitcher_bk_thrown / _total_thrown
        _os_w = pitcher_os_thrown / _total_thrown
    else:
        _fb_w, _bk_w, _os_w = 0.55, 0.25, 0.20
    pitch_mix_barrel_edge = (
        _fb_w * (batter_fb_barrel - pitcher_fb_barrel) +
        _bk_w * (batter_bk_barrel - pitcher_bk_barrel) +
        _os_w * (batter_os_barrel - pitcher_os_barrel)
    )
    pitch_mix_adj = max(-1.5, min(1.5, pitch_mix_barrel_edge * 30))

    plat_score = max(-4.0, min(10.0, 5.0 + (plat_xba - 0.250) * 25 + (plat_hh - 0.360) * 15
                               + k_plat + babip_plat + pitch_mix_adj))
    scores['Platoon'] = (round(plat_score, 1), 10)

    # ── Opponent Defense (0-5, can go negative) ──────────────────────────────
    # Bad defense (more errors) = more hits reach = good for batter
    def_score = max(-3.0, min(5.0, 2.5 + opp_def_rating))
    scores['Opp Defense'] = (round(def_score, 1), 5)

    # ── Pitcher Rest, Workload & GB% (0-5) ──────────────────────────────────
    gb_penalty  = max(-3.0, min(0.0, (0.43 - pitcher_gb_pct) * 10))
    rest_bonus  = max(-3.0, min(3.0, pitcher_rest_factor * 2))
    # High pitch count last start → likely exits earlier → batter gets bullpen exposure
    if pitcher_last_pitch_count >= 110:
        workload_bonus = 1.0
    elif pitcher_last_pitch_count >= 95:
        workload_bonus = 0.5
    else:
        workload_bonus = 0.0
    pitcher_ctx = max(-3.0, min(5.0, 2.5 + rest_bonus + gb_penalty + workload_bonus))
    scores['Pitcher Context'] = (round(pitcher_ctx, 1), 5)

    # ── Team Run Environment (0-5) ───────────────────────────────────────────
    # High-scoring team = more RBI/run opportunities
    team_score = max(0.0, min(5.0, (team_runs_avg - 3.0) / (7.0 - 3.0) * 5.0))
    scores['Team Scoring'] = (round(team_score, 1), 5)

    # ── Batter Rest (-2 to +1) ───────────────────────────────────────────────
    # 0 days = day game after night / doubleheader; 1 = normal; 2-5 = well rested
    if batter_rest_days == 0:
        _brest = -2.0
    elif batter_rest_days == 1:
        _brest = 0.0
    elif batter_rest_days <= 5:
        _brest = 1.0
    else:
        _brest = 0.5  # extended layoff — unknown rust factor
    scores['Batter Rest'] = (round(_brest, 1), 1)

    # ── Bullpen (0-8) ────────────────────────────────────────────────────────
    bp_era_score  = max(0.0, min(5.0, (bp_era - 3.0) / (5.5 - 3.0) * 5.0))
    bp_whip_score = max(0.0, min(3.0, (bp_whip - 1.0) / (1.8 - 1.0) * 3.0))
    scores['Bullpen'] = (round(min(8.0, bp_era_score + bp_whip_score), 1), 8)

    # ── Batted Ball Edge (0-12) — merged barrel + contact quality ────────────
    # Barrel rate edge (pitch-mix weighted)
    barrel_edge = (
        batter_fb_seen * (batter_fb_barrel - pitcher_fb_barrel) +
        batter_bk_seen * (batter_bk_barrel - pitcher_bk_barrel) +
        batter_os_seen * (batter_os_barrel - pitcher_os_barrel)
    )
    # Pitch-mix-weighted whiff matchup
    _total_thrown = pitcher_fb_thrown + pitcher_bk_thrown + pitcher_os_thrown
    if _total_thrown > 0:
        _fb_w = pitcher_fb_thrown / _total_thrown
        _bk_w = pitcher_bk_thrown / _total_thrown
        _os_w = pitcher_os_thrown / _total_thrown
    else:
        _fb_w, _bk_w, _os_w = 0.55, 0.25, 0.20
    _eff_batter_whiff  = batter_whiff_pct_fb * _fb_w + batter_whiff_pct_bk * _bk_w + batter_whiff_pct_os * _os_w
    _eff_pitcher_whiff = pitcher_whiff_pct_fb * _fb_w + pitcher_whiff_pct_bk * _bk_w + pitcher_whiff_pct_os * _os_w
    whiff_adj = max(-2.5, min(1.5, (0.245 - _eff_batter_whiff) * 6 - (_eff_pitcher_whiff - 0.245) * 4))
    xba_edge      = batter_xba - pitcher_xba_allowed
    ev_edge       = (batter_avg_ev - pitcher_avg_ev) / 10.0
    batted_ball_score = max(0.0, min(12.0, 6.0
        + barrel_edge * 60
        + xba_edge * 20
        + ev_edge * 2
        + whiff_adj))
    scores['Batted Ball Edge'] = (round(batted_ball_score, 1), 12)

    # ── Park & Weather (0-9) ────────────────────────────────────────────────
    park_score = max(0.0, min(5.0, (park_factor - 0.90) / (1.15 - 0.90) * 5.0))
    wind_score = max(-2.0, min(2.0, wind_dir * min(wind_speed, 20) / 20 * 2.0))
    temp_score = max(-1.0, min(1.0, (temp_f - 50) / (85 - 50) * 1.0)) if temp_f > 0 else 0
    # Career batter splits at this park — only apply when enough AB history (≥20)
    if park_ab >= 20:
        park_ba_adj  = max(-1.0, min(1.0, (park_ba  - 0.250) * 10))
        park_slg_adj = max(-0.5, min(0.5, (park_slg - 0.400) * 3))
        park_hist    = park_ba_adj + park_slg_adj
    else:
        park_hist = 0.0
    scores['Park & Weather'] = (round(max(0.0, min(9.0, park_score + wind_score + temp_score + park_hist + 1)), 1), 9)

    # ── Batting Order (0-7) — wider spread: leadoff/2-hole=7, bottom order=2
    bo_score = BATTING_ORDER_SCORES[batting_order - 1] if 1 <= batting_order <= 9 else 5
    scores['Batting Order'] = (float(bo_score), 7)

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
        # Line Edge is informational only — not added to score (projection already scored directly)
        pass

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

    # Normalize to 0-100. Several components (hot/cold, home/away, defense, platoon)
    # score 0 for neutral players, so a neutral player scores ~44% of max_possible.
    # We target ~55 for a neutral player: 0.44 * 100 * 1.25 = 55.
    _calib = 1.25
    normalized = (raw_total / max_possible * 100 * _calib) if max_possible > 0 else 0

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
