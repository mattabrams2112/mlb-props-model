"""
The single place that assembles compute_rating's inputs.

`rating.py` scores; this decides what gets fed into it. That decision used to
live twice — `worker._get_rating` and Game View's `get_rating` each hand-built
their own ~80-argument call — and the two drifted, silently, because every
parameter in compute_rating has a default. A caller that forgets one doesn't
error; it scores that component against a plausible-looking constant.

That is not hypothetical. Both features from commit a561e51 (2026-06-02) were
wired into worker.py and not Game View, and neither was noticed for 69 days:

  * park splits    -> park_ab defaulted to 0, and rating.py guards the whole
                      park-history block behind `if park_ab >= 20`, so it
                      contributed exactly zero on the page generating ratings.
  * pitch count    -> defaulted to 0, so the 95/110-pitch workload branch in
                      Pitcher Context never fired.

Three more were found the same day: the worker wasn't passing batter_obp,
batter_sb_rate or batter_rest_days, so it scored Form & Hit Rate and Batter
Rest differently from Game View for the same player.

Add a rating input HERE and both the page and the cron get it at once. There is
a drift guard in `tests/test_rating_parity.py` that fails if either caller
starts building its own call again.
"""
from datetime import datetime

from rating import compute_rating
from weather import get_park_factor
from bvp_stats import get_bvp
from statcast_features import get_batter_statcast, get_pitcher_statcast
from pitcher_data import get_pitcher_season_stats, get_pitcher_last_pitch_count
from park_splits import get_batter_park_splits


def _batter_rest_days(res, game_date):
    """Days since the batter last played. 1 (normal rest) when unknown."""
    try:
        gd = (datetime.strptime(str(game_date)[:10], '%Y-%m-%d').date()
              if game_date else datetime.now().date())
        last = res['df']['date'].max()
        if hasattr(last, 'date'):
            last = last.date()
        return max(0, (gd - last).days)
    except Exception:
        return 1


def score_batter(res, player_id, pitcher_id, park_team, batting_order,
                 temp_f, wind_speed, wind_dir,
                 bp_era=4.20, bp_whip=1.30, is_home=True,
                 line=None, over_odds=None,
                 p_std=None, p_sc=None, p_last3=None, p_rest=None,
                 b_sc=None, team_score=None, ump_data=None, opp_defense=None,
                 pitcher_throws='R', game_date=None, season=None):
    """
    Build every compute_rating input for one batter and score them.

    Source dicts (p_std, p_sc, b_sc, ...) are optional — anything omitted is
    fetched here. The worker already has them from its per-game setup and
    passes them in; Game View lets some be fetched.

    `line` / `over_odds` are label-only in rating.py (Line Edge is displayed,
    never scored), so the worker leaving them None does not change the total.
    """
    season = season or int(res['df']['season'].iloc[-1])

    b_sc  = b_sc  if b_sc  is not None else get_batter_statcast(player_id, season)
    p_sc  = p_sc  if p_sc  is not None else (get_pitcher_statcast(pitcher_id, season) if pitcher_id else {})
    p_std = p_std if p_std is not None else (get_pitcher_season_stats(pitcher_id, season) if pitcher_id else {})
    p_last3     = p_last3     or {}
    p_rest      = p_rest      or {}
    team_score  = team_score  or {}
    ump_data    = ump_data    or {}
    opp_defense = opp_defense or {}
    bvp = get_bvp(player_id, pitcher_id) if pitcher_id else {}

    # Unpacked explicitly rather than **splatted: the helper also returns
    # park_ops, which compute_rating does not accept, and a splat turns that
    # into a TypeError that kills every rating. Naming the three keys keeps the
    # parity test able to check them statically too.
    _park = get_batter_park_splits(player_id, park_team)

    return compute_rating(
        recent_7g         = res['r7g'],
        recent_30g        = res['r30g'],
        season_avg        = res['savg'],
        opp_era           = p_std.get('opp_era', 4.30),
        opp_whip          = p_std.get('opp_whip', 1.28),
        batter_fb_barrel  = b_sc.get('batter_fb_barrel_pct', 0.080),
        batter_bk_barrel  = b_sc.get('batter_bk_barrel_pct', 0.040),
        batter_os_barrel  = b_sc.get('batter_os_barrel_pct', 0.050),
        pitcher_fb_barrel = p_sc.get('pitcher_fb_barrel_pct', 0.080),
        pitcher_bk_barrel = p_sc.get('pitcher_bk_barrel_pct', 0.040),
        pitcher_os_barrel = p_sc.get('pitcher_os_barrel_pct', 0.050),
        batter_fb_seen    = b_sc.get('batter_fb_seen_pct', 0.55),
        batter_bk_seen    = b_sc.get('batter_bk_seen_pct', 0.25),
        batter_os_seen    = b_sc.get('batter_os_seen_pct', 0.20),
        park_factor       = get_park_factor(park_team),
        wind_speed        = wind_speed,
        wind_dir          = wind_dir,
        bvp_avg           = bvp.get('bvp_avg', res['ba30']),
        bvp_sample        = bvp.get('bvp_sample', 0),
        batting_order     = batting_order,
        recent_ba         = res['ba30'],
        temp_f            = temp_f,
        projection        = res['proj'],
        bp_era            = bp_era,
        bp_whip           = bp_whip,
        line              = line,
        over_odds         = over_odds,
        home_hrr          = res.get('home_hrr'),
        away_hrr          = res.get('away_hrr'),
        is_home           = is_home,
        recent_20g        = res.get('r20g_venue'),
        recent_ba_venue   = res.get('ba_venue'),

        batter_hard_hit_pct  = b_sc.get('batter_hard_hit_pct', 0.360),
        pitcher_hard_hit_pct = p_sc.get('pitcher_hard_hit_pct', 0.360),
        batter_xba           = b_sc.get('batter_xba', 0.250),
        pitcher_xba_allowed  = p_sc.get('pitcher_xba_allowed', 0.250),
        batter_avg_ev        = b_sc.get('batter_avg_ev', 88.0),
        pitcher_avg_ev       = p_sc.get('pitcher_avg_ev', 88.0),

        opp_fip                = p_std.get('opp_fip', 4.20),
        opp_last3_era          = p_last3.get('opp_last3_era', 4.30),
        opp_last3_whip         = p_last3.get('opp_last3_whip', 1.28),
        pitcher_throws         = pitcher_throws,
        batter_xba_vs_rhp      = b_sc.get('batter_xba_vs_rhp', 0.250),
        batter_xba_vs_lhp      = b_sc.get('batter_xba_vs_lhp', 0.250),
        batter_hard_hit_vs_rhp = b_sc.get('batter_hard_hit_vs_rhp', 0.360),
        batter_hard_hit_vs_lhp = b_sc.get('batter_hard_hit_vs_lhp', 0.360),
        team_runs_avg          = team_score.get('team_runs_avg', 4.5),
        umpire_tendency        = ump_data.get('umpire_tendency', 0.0),
        opp_def_rating         = opp_defense.get('def_rating', 0.0),
        pitcher_rest_factor    = p_rest.get('rest_factor', 0.0),
        pitcher_gb_pct         = p_sc.get('pitcher_gb_pct', 0.430),

        batter_rest_days = _batter_rest_days(res, game_date),
        batter_obp       = res.get('obp30', 0.320),
        batter_sb_rate   = res.get('sb_rate', 0.05),
        pitcher_last_pitch_count = (get_pitcher_last_pitch_count(pitcher_id, season)
                                    if pitcher_id else 0),
        park_ba  = _park['park_ba'],
        park_slg = _park['park_slg'],
        park_ab  = _park['park_ab'],

        pitcher_fb_thrown    = p_sc.get('pitcher_fb_thrown_pct', 0.55),
        pitcher_bk_thrown    = p_sc.get('pitcher_bk_thrown_pct', 0.25),
        pitcher_os_thrown    = p_sc.get('pitcher_os_thrown_pct', 0.20),
        batter_whiff_pct_fb  = b_sc.get('batter_whiff_pct_fb', 0.200),
        batter_whiff_pct_bk  = b_sc.get('batter_whiff_pct_bk', 0.330),
        batter_whiff_pct_os  = b_sc.get('batter_whiff_pct_os', 0.310),
        pitcher_whiff_pct_fb = p_sc.get('pitcher_whiff_pct_fb', 0.200),
        pitcher_whiff_pct_bk = p_sc.get('pitcher_whiff_pct_bk', 0.330),
        pitcher_whiff_pct_os = p_sc.get('pitcher_whiff_pct_os', 0.310),
        opp_k_pct            = p_sc.get('pitcher_k_pct', 0.222),
        opp_bb_pct           = p_sc.get('pitcher_bb_pct', 0.083),
        opp_babip            = p_sc.get('pitcher_babip', 0.300),
        opp_whiff_pct        = p_sc.get('pitcher_whiff_pct', 0.245),
        opp_k_pct_vs_lhb     = p_sc.get('pitcher_k_pct_vs_lhb', None),
        opp_k_pct_vs_rhb     = p_sc.get('pitcher_k_pct_vs_rhb', None),
        opp_babip_vs_lhb     = p_sc.get('pitcher_babip_vs_lhb', None),
        opp_babip_vs_rhb     = p_sc.get('pitcher_babip_vs_rhb', None),
        batter_k_pct         = b_sc.get('batter_k_pct', 0.222),
        batter_bb_pct        = b_sc.get('batter_bb_pct', 0.083),
        batter_babip         = b_sc.get('batter_babip', 0.300),
        batter_whiff_pct     = b_sc.get('batter_whiff_pct', 0.245),
        batter_k_pct_vs_rhp  = b_sc.get('batter_k_pct_vs_rhp', None),
        batter_k_pct_vs_lhp  = b_sc.get('batter_k_pct_vs_lhp', None),
        batter_babip_vs_rhp  = b_sc.get('batter_babip_vs_rhp', None),
        batter_babip_vs_lhp  = b_sc.get('batter_babip_vs_lhp', None),
    )
