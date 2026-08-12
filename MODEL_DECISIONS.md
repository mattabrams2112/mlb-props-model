# Model Decisions Log

A running record of *why* the rating/projection/betting logic is the way it is —
so decisions don't get re-litigated from scratch. Newest first. Dates are ET.

---

## 2026-08-11a — Calibration disabled. The zeros filter was inverting its sign.

Fix order item 1 from 08-10c. `calibration.py` dropped `actual > 0` before
computing `mean(actual) / mean(projected)`. A 0 is a decided outcome — the
batter went 0-for — so this discarded 32.7% of the sample (5,222 of 15,956
rows), all of it on the losing side.

Measured, live vs corrected:

| tier | n | live factor | correct factor |
|---|---|---|---|
| 90+ | 52 | 0.7484 | 0.4605 |
| 80-89 | 352 | 0.8854 | 0.6681 |
| 70-79 | 1,142 | 0.8517 | 0.6186 |
| 60-69 | 3,118 | 0.9799 | 0.6957 |
| **<60** | **11,292** | **1.4837** | **0.9757** |

In the `<60` tier — 71% of all rows — the filter did not merely shrink the
correction, it **reversed** it. A tier that is very nearly calibrated (0.9757)
was being told it underprojects by 48%. Because worker.py and Game View both
rate in two passes (rate `base_proj`, pick the factor from that rating, re-rate
the corrected projection), the inflated projection was then *re-rated*, which is
a live mechanism for pushing marginal plays up into higher tiers.

**Two changes, deliberately separate in effect:**
1. The zeros filter is removed, so the numbers this module reports are honest.
2. Calibration is switched OFF behind `CALIBRATION_ENABLED = False`.

It stays off because the honest factors are the known-bad ones this log has
described since 08-05: ~0.46 at 90+ collapses the bet band outright. Per
CLAUDE.md, calibration may only be re-enabled together with a re-derived rating
threshold, never alone. `get_correction_factor` is the single choke point for
all three call sites, so the flag is the entire off switch and no caller
changed.

**This changes live scoring** — projections were being multiplied by a non-1.0
factor and now are not. That is intended: the cohort has not opened, and moving
from a sign-inverted correction to no correction is a move toward a known state.
Verified by running a real game (KC @ LAD, 2026-08-11) end to end.

---

## 2026-08-10c — Day zero. Agreed fix order + walk-forward cohort design.

External code review of the post-merge state, verified against the code. Agreed
plan, to execute from 2026-08-11. **Nothing below is done yet.**

**Why today is day zero:** every historical result describes a system that no
longer exists. Park splits scored exactly 0 for 69 days, pitch count never
fired, the cron never once ran the worker, and the projection pipeline existed
in two drifted copies. Prior performance — including the 85-89 band's 63% —
cannot validate the current implementation.

### Fix order

1. **Disable or fix the active calibration.** `calibration.py:78` still filters
   `actual > 0`, which this log recorded on 08-05 as inverting the correction's
   sign. It is applied live at `worker.py:380` and Game View `:433/:453`. One
   line, live impact.
2. **Replace whole-table writes with transactional upserts.** `save_all` does
   `to_sql(if_exists='replace')` on every call — once per batter — while the web
   app and cron write concurrently. Read-modify-replace means the later writer
   silently erases the earlier one. Needs permanent tables with unique
   constraints and `INSERT ... ON CONFLICT DO UPDATE`. Silent corruption, no
   error surface; highest actual risk on the list.
3. **Correctness batch.**
   - Empty-frame `.any()` crash at `worker.py:107` and `game_pred_engine.py:71`
     (`not df.empty and ...` short-circuits to a bool). Same bug already fixed
     in `full_tracker.py`; the siblings were missed.
   - `worker.py:303` calls `get_cached_rating(game_date, pid)` without the
     pitcher, so a doubleheader's Game 2 can inherit Game 1's rating. Game View
     passes it correctly.
   - Adopt MLB `game_pk` as the durable game identity. Fixes both
     `worker.py:450`'s `gid = f'{away}_{home}'` colliding across doubleheaders
     AND the play log's ~0.93% duplicate rows from probable-pitcher changes.
   - Declare pytest in requirements; add CI. The pre-push hook is untracked and
     therefore machine-local — currently the only automated gate.
4. **Modeling.**
   - **Synthetic future prediction row.** `projection.py` uses the last
     COMPLETED game as the prediction template, and its `shift(1)` features
     exclude that game's own result — so every projection is one game stale.
     Append a synthetic future row whose features run through the latest
     completed game. NOTE: the rating's `r7g`/`r30g` come from `df.tail(n)` and
     DO include the latest game, so staleness is confined to the XGB/LGBM input.
   - **Historical pitcher-feature leakage.** `get_pitcher_statcast(pid, season)`
     returns season aggregates applied to every historical training row, and
     rolling stats are blended with full-season ERA/FIP. Training rows therefore
     see data recorded after those games. This inflates apparent model quality,
     which is invisible from results — the most important modeling item here.

**Deliberately NOT changing:** the pandas bound stays `<3.1.0`. The review reads
this as permitting "the version that broke production," but pandas 3.0 broke it
only via the `get_rolling_pitcher_stats` bug fixed on 08-10, and the full worker
has since been run end-to-end on pandas 3.0.3 locally and in production.
Pinning `<3.0.0` would revert to a configuration untested since the fix.

### Walk-forward cohort

Freeze code and configuration, stamp a **model version (git SHA) on every
prediction**, pre-register metrics and thresholds here, and do not adjust the
model until the window closes.

**The probability diagnostic is a composition of two components:**

    projected mean  ->  probability mapping  ->  comparison with market

**Both must be frozen and versioned before any observation counts.** Changing
either one invalidates the cohort, because a shift in the market comparison
cannot be attributed without knowing which half moved. This is why replacing the
Poisson mapping (below) is a cohort-ENTRY requirement rather than an improvement
to make later — a cohort opened on a mapping known to be misspecified measures
the mapping, not the model.

**Power analysis — why W/L cannot be the only short-term diagnostic.** At
approximately 1.1 bet-band plays per day, detecting a true 57% win rate against
52.4% breakeven requires roughly **724 plays — about 1.8 years — assuming a
one-sided 5% significance level and 80% power. A two-sided test would require
approximately 920 plays.** Consequently, bet-band W/L cannot be the model's only
short-term diagnostic, and the existing band results remain statistically noisy.

| true win rate | plays (1-sided, 80% power) | time at 1.1/day |
|---|---|---|
| 57% | 724 | ~1.8 years |
| 60% | 263 | ~0.7 years |
| 63% | 134 | ~0.3 years |

`novig_prob` began recording 2026-08-10 for every lined play — approximately 80
per day, or 2,400 per month. This supplies a high-volume, pre-game market
benchmark for measuring model-market disagreement, ranking behaviour, and
stability within weeks. **Uncertainty will be calculated with clustering by game
or day, because player props from the same slate are not independent** — an
earlier draft of this entry asserted SE ≈ 0.02 with no justification; the naïve
figure for 2,400 Bernoulli observations near 50% is about 0.010, and clustering
will widen it by an amount to be measured rather than guessed.

This benchmark accelerates diagnosis, but **only subsequent outcomes can
establish calibration, predictive value, and betting profitability.** The market
comparison can quickly reveal disagreement, instability, ranking problems and
implausible calibration; it cannot prove edge. W/L remains the ultimate
profitability test — it is simply too slow to be the only one.

**Pre-registered metrics** (fill thresholds in before the window opens). Note
the split between *what settles the question* and *what gives early warning* —
they are different jobs and must not be conflated:

*Deciding metrics — outcome-based, slow, definitive:*
- 85+ win rate vs the 52.4% breakeven. Accumulates over seasons, not weeks;
  report the confidence interval every time, never the bare percentage.
- Brier score and log-loss of model probability against realised outcomes,
  benchmarked against `novig_prob` scored the same way. Beating the market here
  on a large sample is the real claim; nothing else substitutes for it.
- Projection MAE vs the two baselines that already beat it — flat 1.77
  (MAE 1.479) and `r30g` (1.492). The pipeline must clear both.

*Diagnostic metrics — fast, high-volume, cannot establish edge:*
- Model-vs-market disagreement: calibration slope and intercept of model
  probability against `novig_prob`, with game- or day-clustered uncertainty.
- Ranking behaviour: does model rank order track market rank order, and where
  does it diverge most.
- Stability: do these move between slates, which would indicate the pipeline
  is not deterministic.
- *Guardrail:* per-bin `actual` vs `base_proj` must stop being flat. On 9,134
  rows it ran 1.61 at base_proj 0.7 rising only to 2.22 at 4.65.

A diagnostic going wrong is grounds to stop and investigate. A diagnostic
looking good is **not** grounds to claim edge.

**Cohort rules:** correctness fixes that provably do not change scoring may land
mid-window and must be logged. Anything touching scoring resets the cohort and
starts a new version. No threshold or tier changes until the window closes.

### Pre-window blocker: the Poisson probability mapping

`odds_api.fair_probability` maps a projection to P(over) with **Poisson**. HRR is
structurally non-Poisson: a home run contributes at least three units at once
(a hit, a run, an RBI), so the components are strongly positively correlated and
variance must exceed the mean. This has to be settled BEFORE the cohort opens,
because a mis-specified mapping means the market diagnostic measures the
distributional assumption as much as it measures the model.

**Conclusion: narrow conditional bins exhibit stable structural overdispersion
and excess zeros relative to Poisson, strongly establishing that the current
Poisson probability mapping is unsuitable.** Binned by `r30g` — a baseline the
model never touches, so the mixture-overdispersion artifact is avoided:

| r30g bin | n | mean | var | var/mean | P0 obs | P0 Poisson |
|---|---|---|---|---|---|---|
| 0.0-1.25 | 482 | 1.45 | 2.95 | 2.04 | 0.382 | 0.235 |
| 1.5-1.75 | 630 | 1.71 | 3.20 | 1.87 | 0.317 | 0.180 |
| 1.75-2.0 | 864 | 1.73 | 3.54 | 2.04 | 0.322 | 0.177 |
| 2.25-2.5 | 391 | 1.87 | 3.82 | 2.04 | 0.307 | 0.153 |
| 2.5-3.0 | 213 | 1.91 | 3.99 | 2.09 | 0.291 | 0.149 |

Poisson requires var/mean = 1.00. Observed is ~2.0 in every bin, and observed
zero rates run 1.7-2x Poisson throughout.

**A preliminary line-level calculation suggests approximately 7 percentage
points of average overstatement at line 1.5, but that magnitude remains subject
to conditional-bin weighting and time-separated validation.** Status so far, all
at line = 1.5:

| | n | empirical | Poisson | NB1 |
|---|---|---|---|---|
| all, pooled mean | 1,957 | 0.432 | 0.504 (+0.072) | — |
| all, bin-weighted | 1,957 | 0.432 | 0.503 (+0.071) | 0.428 (-0.004) |
| later-half split | 1,086 | 0.442 | 0.505 (+0.063) | 0.430 (-0.012) |

Bin-weighting moved the estimate by ~0.001: the conditional means cluster in
1.45-1.91 and the Poisson tail is near-linear across that range, so Jensen's
inequality contributes little *here*. That is a measured result, not an
assumption, and it may not hold at other lines.

**Parameterization, recorded precisely.** A stable var/mean ≈ 2 implies an
**NB1**-style relationship, `Var(Y|mu) = 2*mu`. In the standard negative
binomial parameterization that is **fixed p = 0.5 with size r = mu** — the size
varies with the conditional mean. It is *not* a single fixed-size NB2 model.

**Still required before freezing a mapping:**
1. Bin-level Poisson/NB probabilities weighted by sample counts, per line — done
   for 1.5 only; 0.5 has n=214 and other lines are too thin.
2. A genuine time-separated holdout. The split above validates the functional
   form out-of-sample but **the dispersion parameter was estimated on the full
   sample including the holdout half** — re-estimate on the earlier period only.
3. Evaluate candidates on log loss and Brier, not calibration slope alone.
4. Compare NB1, empirical line-specific calibration, and a direct binary
   probability model. Caution on the last: it conditions on a line existing,
   which drops ~25% of plays and risks learning the market rather than the sport.
5. Freeze the selected mapping before the forward cohort opens.

**Scope of the claim.** These probabilities were computed from the *empirical*
mean per bin, not from model projections, so they establish the distributional
shape given a correct mean — they say nothing about the model's ability to
produce that mean, which remains the separate and weaker problem. And because
historical projections came from an obsolete system, this analysis can show
Poisson is structurally unsuitable but **cannot** establish that any selected
mapping is calibrated for today's model.

**Consequence to record:** the historical 1.5-line population appears inflated
by approximately 7pp on average. Per-play bias varies with the conditional mean.
Breakeven at -110 is 52.4%; the mapping asserted ~50.4% where the realised rate
was 43.2%. This is a second, independent source of Edge inflation, compounding
with the projection bias documented on 08-05.

---

## 2026-08-10b — One pipeline; and the two bugs that were actually zeroing the worker

**Merged the duplicated pipeline into `projection.py`.** Game View and worker
each had their own copy; both now import one. The merged version keeps the
worker's modelling and Game View inherits all of it: real pitcher history
(`fast_mode=False`), pitcher features in training, recency weighting (exp decay
over 90 days), and power/contact hyperparameter tuning.

**Pitcher multiplier switched OFF** (`APPLY_PITCHER_MULTIPLIER`). With real
pitcher features in training it counts the opposing starter twice, on top of the
five rating components that already read pitcher. The 8/05 diagnostic had
already measured a ~2x over-response — corr(opp_era, projected) 0.263 vs
corr(opp_era, actual) 0.142. The function is kept intact; it is one constant.

**Root cause of "everything is messed up" — two separate bugs, neither of them
the ceiling drift.** Reproduced locally by installing the full dependency set
and running the worker against a real slate.

1. **`pitcher_data.get_rolling_pitcher_stats` broke under pandas 3.**
   `cutoff = ... if not isinstance(game_date, type(pd.Timestamp(0).date())) else game_date`
   — `pd.Timestamp` subclasses `datetime`, which subclasses `date`, so the
   isinstance test was always True for the Timestamps this is called with and
   the conversion was skipped, leaving `cutoff` a Timestamp. pandas 2 tolerated
   comparing that against `.dt.date` values; **pandas 3 raises TypeError**.
   `requirements.txt` had `pandas>=2.0.0` unbounded, so the rebuild triggered by
   today's deploy pulled pandas 3.0.5 and every projection in the worker began
   failing — it uses `fast_mode=False`, which is the only path that reaches this
   function. Game View was unaffected only because it still ran `fast_mode=True`;
   merging the pipelines would have broken it too. Fixed, and pandas/numpy now
   carry upper bounds.

2. **`lineup_fetcher`'s boxscore fallback never populated batting-order codes.**
   It set `home_batters` and `lineups_official = True` but left
   `home_batter_codes` empty. The worker identifies starters solely via
   `ocode % 100 == 0`, so an empty code map means every batter fails the starter
   test and the slate scores **zero** players. It only bites in the pre-game
   window where MLB has published a boxscore lineup but not yet the schedule
   `lineups` hydration — exactly the window the worker exists to fill. On
   today's slate this took the worker from 0 startable batters to 45. Pre-existing
   and unrelated to any change here.

Also fixed: `full_tracker.log_play` crashed on a completely empty log
(`not df.empty and ...` short-circuits to a bool, which has no `.any()`).

**Verified end-to-end**, worker against BOS @ TOR: 18 batters scored in 102s
with warm caches, `base_proj` / `proj_ceiling` / `proj_src` / `r30g` all
populating, ceiling cap confirmed active (Guerrero base 4.33 -> capped 3.87).

**Watch item:** 102s per game is fine for a 30-minute cron but slow for a live
page. Game View caches per game for 24h, so the cost is a one-off first load —
provided `DATA_DIR` points at a mounted volume. If it does not, the pitcher
cache is wiped every deploy and every first load pays full price.

**Correction to the entry below:** it states the worker's stale 1.8 ceiling was
freezing bet ratings. The mechanism is real, but it was never established that
it drove the 8/03 win-rate drop, and the drop is better explained by n=10. Read
that entry as a code-level defect, not a diagnosis.

---

## 2026-08-10 — There were TWO models. The worker never got the 8/03 change.

Investigating "the model feels off since 7/22". The premise turned out to be
wrong and the real defect was structural.

**The 7/22 changes are not when win% dropped.** Real-line plays, 85+:

| window | n | win% |
|--------|---|------|
| 6/18-7/21 (pre-7/22) | 67 | 56.7% |
| 7/22-8/02 | 7 | 71.4% |
| **8/03-8/09** | **10** | **20.0%** |
| population, all three windows | — | 45.2 / 43.6 / 44.4 |

The 7/22 window went UP. The drop is entirely post-**8/03** — the ceiling
loosening. Population win% is flat across all three, so nothing global broke in
grading or data. P(<=2 wins in 10 | true 56.7%) = **2.1%**: low end of variance,
not proof.

**Finding 1 — `worker.py` and Game View were two different models.**
Both score every batter; both write `ratings_cache`; `save_rating` is
first-write-wins (`ratings_cache.py:106`) and Game View honors the frozen value
without recomputing. The worker runs on a 30-minute cron, so it frequently wins
that race. It differed from Game View in three ways:

| | worker (before) | Game View |
|---|---|---|
| projection ceiling | **1.8x** | 2.0x |
| pitcher-quality multiplier | **absent** | applied |
| lineup-context adjustment | **absent** | +/-12% |

So **the 8/03 loosening to 2.0 never took effect for any play the worker rated
first** — it was only ever applied to `pages/2_🎯_Game_View.py:179`. The 8/03
evaluation has therefore been measuring a blend of two models, and its explicit
revert trigger cannot be read until this is fixed.

The worker also rated the RAW projection while logging a CALIBRATED one, so its
rating and the projection displayed beside it described different numbers.

**Fix:** ported Game View's ceiling, pitcher multiplier, lineup context, and
two-pass calibration ordering into the worker. Prediction and rating are now two
pooled passes, because lineup context needs the whole lineup before any single
batter can be rated. Cached batters contribute their frozen projection to that
context (post- vs pre-calibration, so a proxy) — otherwise later worker passes,
where most of the lineup is cached, would average two batters; below 5 known
spots the context stays neutral. Constants verified identical across both files.

**Not done:** the two copies still exist. Deleting one is the real fix and is
deliberately deferred — too invasive mid-season.

**Finding 2 — `base_proj` was being erased exactly where it matters.**
`log_play`'s update path blanked `base_proj` whenever a caller didn't supply one.
`save_rating` mirrors to `full_play_log` without it, and that mirror **only fires
for plays that qualify as bets** — so the loss was concentrated in the bet band:

| rating band | rows | base_proj missing |
|---|---|---|
| <70 | 3,515 | **0.0%** |
| 80-85 | 32 | 43.8% |
| **85-90** | **11** | **90.9%** |

Now only overwritten when actually provided.

**Finding 3 — the no-vig benchmark shipped 8/05 never recorded anything.**
`get_hrr_lines` computes it correctly; `get_player_line` rebuilds the return dict
by hand and dropped the key. **0 of 872 rows** since 8/06 had it despite 642
having a real line. Fixed.

**Finding 4 — calibration is still NOT ready to ship, for a new reason.**
Fitting was never the blocker: 7/22->now has 3,698 decided rows with `base_proj`
and the per-bin multipliers match the full season within ~0.06. Two real blockers:

1. Fitted on the correct (pre-calibration) denominator, the bias inverts at the
   bottom — `<60` UNDER-projects at 1.08, not the 0.76 the 8/05 entry reported
   from post-calibration `projected`. That entry's tier table should not be used.
2. `actual` is FLAT — 1.61 at base_proj 0.7 rising only to 2.22 at 4.65. Honest
   calibration is "predict ~1.8 for everyone," which reproduces the 8/05 Finding
   3c on 9,134 clean rows. Installing it collapses `proj_score` to a constant,
   costs 80-89 plays ~6 rating points, and empties the band.

Bin on `base_proj`, not rating tier: the rating-keyed fit is circular (looked up
by pre-calibration rating, fitted on post-calibration rating) and unstable —
80-89 swings 0.92 -> 0.57 across halves where every base_proj bin holds to ~0.1.

**Gate unchanged:** ~20-25 decided 85-89 plays post-8/03. Currently **7**. That
count now restarts in practice, since plays before today were rated by a blend
of two models.

**Also logged now:** `proj_ceiling` (so the 1.8 vs 2.0 evaluation can separate
plays the old cap would have evicted from organic ones) and `proj_src`
('worker' / 'gv', so a fit can prove which path produced a row rather than
assume).

---

## 2026-08-05 — Full-DB diagnostic: the RATING works, the PROJECTION is broken

First analysis run against the live Postgres play log (14,935 rows; 8,228 decided
plays since the 2026-06-18 model date) rather than reconstructed samples.

**Finding 1 — `calibration.py`'s `actual > 0` filter inverts the answer.**
4,781 of 14,656 decided plays (32.6%) have `actual = 0`. Dropping them doesn't
just weaken calibration, it flips its sign:

| sample | multiplier | what it tells the model |
|--------|-----------|--------------------------|
| all decided plays (truth) | **0.757** | over-projects by 32% |
| `actual > 0` (what runs today) | **1.121** | *under*-projects by 12% |

At `<60` it applies **1.31**, actively inflating projections that are already high.
At 80-89 it applies 0.82 where truth is 0.62.

**Finding 2 — over-projection scales monotonically with rating.**

| tier | n | proj | actual | true mult |
|------|----|------|--------|-----------|
| 90+ | 23 | 4.374 | 1.565 | 0.358 |
| 80-89 | 156 | 4.008 | 2.487 | 0.621 |
| 70-79 | 580 | 3.295 | 1.895 | 0.575 |
| 60-69 | 1929 | 2.963 | 1.982 | 0.669 |
| <60 | 5540 | 1.964 | 1.664 | 0.847 |

The broad population is only ~18% high; 90+ projects ~2.8x reality. **This
supersedes the 2026-07-22 "recenter by a constant offset" plan** — the bias is
multiplicative and rating-dependent, so a constant subtraction cannot fix it.

**Finding 3 — projection lift above a player's own baseline is pure noise.**
Bucketing by `projected / r30g`, `actual` is FLAT while projection climbs:

| proj/r30g | n | proj | actual |
|-----------|----|------|--------|
| <1.00 | 851 | 1.298 | 1.673 |
| 1.25-1.50 | 354 | 2.537 | 1.641 |
| 1.50-2.00 | 727 | 3.061 | 1.744 |
| 2.00-3.00 | 237 | 2.990 | 1.658 |

Everything the multiplier stack adds above `r30g` moves the projection and not the
outcome.

**Finding 3a — the "pitcher counted 3x" theory was TESTED AND REFUTED.**
Initial hypothesis: opposing-starter quality reaches the projection three times
(`_pitcher_mult` in run_prediction, `Starter Matchup` via `matchup_pct`, and
indirectly via `_ctx_pct`, since teammate projections already carry
`_pitcher_mult`) — modelled compounding x0.58 vs an elite SP, x1.55 vs a bad one.
The triple path is real in the code, but it is NOT the driver. Joining season ERA
for all 239 opposing starters onto all 8,228 decided plays:

| opp ERA | n | proj | actual | over | mult |
|---------|----|------|--------|------|------|
| 0.0-3.0 | 1333 | 1.877 | 1.344 | +0.533 | 0.716 |
| 3.5-4.0 | 1897 | 2.254 | 1.651 | +0.603 | 0.732 |
| 4.5-5.0 | 976 | 2.604 | 1.961 | +0.642 | 0.753 |
| 5.0-6.0 | 1105 | 2.800 | 2.083 | +0.717 | 0.744 |
| 6.0+ | 650 | 2.846 | 2.235 | +0.610 | 0.786 |

**corr(opp_era, over-projection) = 0.0157** — flat. The bias is uniform across
matchup quality, not compounding on bad pitchers. The pitcher signal itself is
real and correctly aimed: corr(opp_era, projected) = 0.263 vs corr(opp_era,
actual) = 0.142, i.e. the model over-responds to pitcher by roughly 2x, but that
is mild over-weighting and not the source of a 33% inflation. Do NOT de-duplicate
the pitcher path expecting it to fix over-projection.

**Finding 3b — the real driver is recent-form extrapolation.**
Over-projection scales with the batter's OWN 30-game form, which the pitcher cut
ruled out:

| r30g | n | proj | actual | mult |
|------|----|------|--------|------|
| 0.0-1.0 | 80 | 1.292 | 1.113 | 0.861 |
| 1.5-2.0 | 1058 | 2.251 | 1.715 | 0.762 |
| 2.0-2.5 | 687 | 2.593 | 1.789 | 0.690 |
| 2.5-3.0 | 165 | 2.731 | 1.903 | 0.697 |

The model extrapolates hot streaks instead of regressing them to the mean.

**Finding 3c — the projection loses to a constant.**

| forecast | MAE | corr w/ actual |
|----------|-----|----------------|
| model projection | 1.7186 | 0.058 |
| `r30g` alone | 1.4922 | 0.084 |
| **flat 1.77 for everyone** | **1.4791** | — |

Predicting one number for every player beats the whole pipeline. Caveats: MAE
rewards central predictions and single-game HRR is near-unpredictable (best MAE
~1.48 by any method), and the r30g sample is 2,599 rows. But the correlation
comparison points the same way — the pipeline adds nothing over `r30g`.

**Finding 3d — why the rating still works anyway.**
`proj_score = min(25, projection/3.5*25)` saturates at proj 3.5. Share of plays
already saturated: 80-84 **61.5%**, 85-89 **68.4%**, 90+ **70.8%** (vs 8% below
60). For most bet-band plays the Projection component contributes a flat 25 and
does no ranking at all — the discrimination comes from the other 117 raw points.
The cap is effectively shielding the rating from the broken projection. This is
also why the band still wins 63% despite Findings 1-3c.

**Finding 4 — the rating itself is sound and profitable.**

| band | n | actual HRR | win% vs real line |
|------|----|-----------|-------------------|
| 80-84 | 101 | 2.465 | 50.0% |
| **85-89** | **55** | **2.527** | **63.0%** |
| 90+ | 23 | 1.565 | 40.9% |
| population | — | 1.770 | 44.9% |

Breakeven is 52.4%. 85-89 at 63% over 54 plays validates the threshold, the 90-94
fade, and the 80-84 drop. It also settles the 8/05 scare that 85-89 had gone 0-2
this week: noise against a 63% base.

**DECISION: change nothing on the rating/projection path right now.**
`Projection` is the largest rating component and saturates at proj 3.5; 80-89
plays average 4.008, pinned at the 25-point cap. Correcting the projection to its
true ~2.5 drops that component ~7.1 raw points ~= **-6.3 rating points**, turning
an 87 into an 81 and emptying the band that currently wins 63%. The 85+ threshold
is calibrated *to the inflated projection*. Fixing the projection is a rescale,
not a bug fix, and must be paired with re-deriving the threshold from data.
Deferred until the 8/03 ceiling evaluation finishes (needs ~20-25 decided plays;
had 2 as of 8/05).

**Consequence:** Edge and Fair Odds are unreliable today (projection inflated up
to ~2.8x at the top). Do not make decisions from the Edge column. Bet selection
reads the rating, which is unaffected.

**Shipped today (deliberately chosen because neither touches ratings):**
1. **Retractable-roof fix** — `DOMED = {'TB','TOR','TEX','MIA'}` was wrong both
   ways: TOR/TEX/MIA were pinned closed (never got weather on open-roof days) and
   HOU/SEA/ARI/MIL were absent entirely (got live wind/temp with the roof shut).
   Verified on 2026-08-04: HOU and ARI were both "Roof Closed" while being scored
   with live wind. Now reads MLB's per-game `gameData.weather.condition`; unknown
   resolves to neutral, since a wrong wind reading corrupts Park & Weather whereas
   neutral merely adds nothing. Shifts ratings <=~1.7 points at those 7 parks.
2. **Market implied team run total logged** (`team_total`) — nothing in the model
   fetched Vegas totals. Runs and RBI are 2 of the 3 HRR components, so expected
   team runs is the most direct driver of the stat, and the model's only view of
   it was `team_runs_avg`, a SEASON average that is identical every day. The
   market number is game-specific and already prices the starter, park, weather,
   bullpen, lineup and late scratches. Derived as `total/2 +/- margin/2` with the
   margin from the devigged moneyline (`z(p) * sigma`). **sigma=4.0 validated
   empirically**, not guessed: real MLB finals 6/18->8/4 give a run-margin sd of
   4.056 (mean +0.130) and mean game total 8.796. **Benchmark only — not wired
   into any rating component.** Also note `umpire_tendency` is a dead parameter:
   fetched per game, passed into `compute_rating`, and never used in any score.
3. **No-vig market probability logged** — `odds_api` only ever read the `Over`
   outcome, discarding the `Under` from the same response, so the book's margin
   could never be removed. Now pairs both sides within a single book, strips the
   vig, and stores consensus `novig_prob` on every play. **Benchmark only — never
   feeds ratings, projections, or bet selection.** Gives a calibration target
   available pre-game, so projection bias becomes measurable in days across
   hundreds of plays instead of waiting on decided outcomes.

---

## 2026-08-03 — Projection ceiling loosened 1.8x -> 2.0x (volume starvation)

**What:** `PROJ_CEILING_MULT` in Game View's `run_prediction` raised 1.8 -> 2.0.

**Why:** at 1.8 the cap over-suppressed the top of the scale. In the 12 days
since the cap went in (7/22 -> 8/3) the Analytics band table showed:
- **85-89: only 6 decided plays** (~0.5/day, down from ~1.5/day pre-cap)
- **95+: ZERO plays**
- 90-94: 1 play (faded anyway)

That's not selectivity, it's starvation — there was no volume left to even
measure whether the band still performs. The mechanism: the cap pulls a
matchup-inflated projection down, which lowers the rating's projection
component (0-25, saturates at proj 3.5), which drops plays out of 85-89.

**Why 2.0 is still safe:** it kills the fantasy projections that started all
this (a 1.2-recent-form hitter caps at 2.4, not the 5.09 we saw), so the Edge
column stays honest. It only admits borderline plays back.

**Honest note:** the multiplier was ALWAYS an arbitrary guess (see the cap's
introduction) — it's a volume/quality knob, not a fitted number. Tuning it for
volume preference is not overfitting to results the way changing it on 6 plays
of W/L would be. Deliberately NOT touching the 90-94 fade or the 80-84 drop.

**How to judge it:** watch whether the newly-admitted 85-89 plays win at a
reasonable rate. Holds ~60%+ at better volume -> keep 2.0. Clearly drags the
band down -> revert to 1.8 (one constant). Superseded entirely once calibration
lands, which centers projections properly instead of clipping them.

**Decision confirmed 2026-08-03: HOLD at 2.0 and evaluate in ~2 weeks.**
Considered objection: "an 84 that wasn't hitting will now show as an 85." Partly
valid — the marginal plays entering the band ARE below the band's average, so
expect the 85-89 rate to come down from its (n=6) high. But those plays were not
organically 84s; the 1.8 cap was *evicting* plays that qualified as 85-89 under
the original settings (part of the 47-play / 61.7% sample). Also note the 27-play
/ 44.4% 80-84 figure is a BLEND of genuinely-mediocre plays and the good ones the
cap knocked down — it does not establish that the evicted plays were bad.

Trade accepted: ~60-65% on ~1-1.5 plays/day (measurable, above the 55.6%
breakeven) over ~70% on ~0.5 plays/day (unmeasurable — cannot distinguish 70%
from 50% at n=6).

**Explicit revert trigger:** if the newly-admitted 85-89 plays hit below ~55%,
revert to 1.8 immediately. Evaluate via the band diagnostic once 85-89 has
~20-25 decided plays.

**Supporting data from the same 7/22->8/3 table:** 80-84 at 27 plays / 44.4%
(proj>=1.5) — validates the 7/27 drop of that tier with real volume. 70-74 at
94 plays / 30.9% — badly inverted, but unbet, so filed not fixed.

---

## 2026-07-27 — 80-84 tier dropped again (poor early results, small sample)

Pulled 80-84 from tracked bets from 2026-07-27 forward — it looked bad over its
short live window. `bet_config.TIER2_END = 2026-07-27`; 80-84 now counts only in
`[EXPANSION_DATE, TIER2_END)` = 7/21–7/26. The 6-day history is KEPT (documents
the experiment); forward is untracked. 85-89 and 95+ unchanged.

**Honest caveat:** this is ~6 days / a small sample, decided on results not a
diagnosed mechanism (unlike the 90-94 fade). Could be variance. Justified because
80-84 was always the marginal, half-staked *experimental* tier — cheap to pull,
trivial to re-add (just move/remove TIER2_END) if a larger clean sample later
shows it's fine. Worth re-checking the 80-84 band diagnostic on the clean window
before any re-add.

---

## 2026-07-22 — Edge diagnostic: recenter, not rebuild (directional, pending clean data)

**What:** Before rebuilding the projection formula, tested the "Edge is fake" story
against outcomes (Edge diagnostic, Analytics). Real-line decided plays, Jun 18→now.

**Check 2 — does higher Edge win?** Yes, directionally:
| edge | win% |
|------|------|
| <0   | 40.2 |
| 0–0.5| 46.3 |
| 0.5–1| 45.6 |
| 1–1.5| 46.5 |
| 1.5–2| 50.4 |
| 2+   | 49.2 |
Negative-edge plays win 40% vs 46-50% for positive edge → the projection **ranks
plays correctly**, it's just shifted up. That's the RECENTER (constant-offset) case,
not the noise/rebuild case. Magnitude signal is weak (positive buckets ~flat) and
the 2+ bucket rolls off — extreme edges are the over-inflated stacked plays.

**Check 1 — projection flat while rating swings?** Partly. Rating swings hard with
the line (41→65); projection rises too (1.86→2.8) but less and plateaus. And the
broad-population projections (1.86–2.8) sit CLOSE to actuals (1.58–2.18), gap ~0.5.
**Correction to an earlier claim:** "projections are ~2 HRR high everywhere" was
overstated — the severe over-projection is CONCENTRATED in the high bet-band,
stacked-matchup boom-or-bust plays; the broad population is only mildly high (~0.5).

**Caveat:** the 85+ real-line sample (the plays we actually bet) is thin here —
lines weren't pulling for high plays — so this is DIRECTIONAL on the mechanism,
not proven on the bet bands. Gated on clean forward data before shipping.

**Plan (do both once clean data confirms):**
1. **Recenter / calibration** — subtract the constant offset (the cheaper fix the
   data supports), NOT the bigger matchup-response rebuild. Rewrite calibration.py:
   drop the `actual > 0` filter, bin by projection, clean-window data, auto-activate
   on sample.
2. **Negative-edge filter** — skip projection < line plays, layered ON TOP of the
   85+ rating filter (not standalone — the broad population never clears breakeven).

---

## 2026-07-22 — Boom-or-bust penalty: tried, then removed (UNSUPPORTED, not disproven)

**What:** Added then removed a rating penalty that docked plays whose projection
sat far above the batter's own recent baseline (`boom_delta = projected − r30g`),
on the theory that these are matchup-stacked "boom-or-bust" spots that bust to 0.

**Why removed:** The `boom_delta` reconstruction (Analytics → API-pull view) over
the 90-94 band, June 18→Jul 22, **n=17**, window=10:
- Wins avg boom_delta **+1.72**, losses **+0.95**, separation (L−W) **−0.77**.
- Sorted by boom_delta the W/L rows were fully interleaved (W,W,L,W,L,W,L,L,L,W) —
  no knee. The two most over-projected plays were both **wins**.
- So the signal is not predictive **at this sample**, and worse, mildly backwards.

**Three-window confirmation (baseline window sensitivity):** re-ran the
reconstruction at windows 10 / 20 / 30 to rule out a too-noisy short baseline —
| window | wins | losses | separation (L−W) |
|--------|------|--------|------------------|
| 10     | +1.72| +0.95  | −0.77            |
| 20     | +2.31| +1.66  | −0.64            |
| 30     | +2.43| +1.83  | −0.61            |
All three are weak AND inverted (wins more over-projected than losses), and the
separation converges to ~−0.6 rather than trending toward a flip. Table stays
interleaved at every window. Cross-check: Corey Seager's baseline went 1.9 (w10)
→ 1.4 (w20), i.e. a real multi-week slump, not a short-window blip — so the
baselines are stable and the null result is not a windowing artifact.

**Important caveat:** 17 plays reconstructed from API calls is still thin. Read
this as **"unsupported across windows at n=17,"** NOT "proven wrong forever."

**Why pulled from production anyway:** an unvalidated penalty was actively shaving
points off live 85-89 / 95+ bets. Testing a hypothesis and betting real units on
it are different bars — remove from prod, keep testing.

**Re-test path:** `r30g` is now logged with every play (clean, live 30g HRR
baseline, no leakage — unlike the polluted historical actuals). In a few weeks
there's a much larger clean sample to re-check whether `boom_delta` (or a variant:
baseline *level* rather than the gap; longer windows; ratio vs difference) has
signal. One faint lead from the n=17 read: **wins had slightly higher real recent
form (~3.0 vs ~2.6 baseline)** — i.e. the baseline itself, not the gap, may matter.

**Kept (independent of this hypothesis):**
- Projection cap (re-cap final projection at the player's realistic ceiling) —
  about honest Edge numbers, not W/L prediction. Still valid.
- 90-94 fade — strengthened by this result: no `boom_delta` filter rescues the
  good 90-94 plays, so the blanket fade is the honest call, not a crude one.

---

## 2026-07-22 — Drop 90-94 from tracked bets (keep 95+)

Diagnostics (band diagnostic + three time windows) showed 90-94 over-projects
~2.5 HRR and wins only ~41%, driven by matchup-stacking selecting boom-or-bust
hitters. From `CAP_DATE` (2026-07-22) forward, ratings in [90, 95) are no longer
tracked bets; 95+ is kept. Date-gated in `bet_config.py` — days before CAP_DATE
keep their 90-94 plays unchanged.

## 2026-07-21 — Add 80-84 tier at 0.5u

80-84 tracked as bets at 0.5u ($4) from `EXPANSION_DATE` (2026-07-21). 85+ stays
1u. Date-gated so prior days remain 85+-only and past records don't move.
Central config in `bet_config.py`.

## ~2026-07 — Bet threshold 85+, flat 1u, strict real-line grading

Raised qualifying rating to 85+, flat 1u staking, and stopped auto-grading no-line
plays against a fake 1.5 (that inflated win rates). Analytics/Weekly still grade
no-line plays vs a 1.5 benchmark for research only (never written to the DB).

## 2026-06-18 — Current projection model

Pitcher multiplier reweighted to ERA 28% / FIP 22% / WHIP 20% / K% 18% / BB% 12%
(+ BvP when ≥10 AB). This is the current model — evaluate the model against data
from this date forward.
