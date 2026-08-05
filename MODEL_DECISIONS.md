# Model Decisions Log

A running record of *why* the rating/projection/betting logic is the way it is —
so decisions don't get re-litigated from scratch. Newest first. Dates are ET.

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
outcome. Suspected cause: opposing-starter quality is applied to the projection
three times — `_pitcher_mult` (run_prediction), `Starter Matchup` via
`matchup_pct`, and indirectly through `_ctx_pct` (teammate projections already
carry `_pitcher_mult`). Modelled compounding: x0.58 vs an elite SP, x1.55 vs a bad
one. Mechanism is verified in code; that it *causes* Finding 3 is still inference.

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
2. **No-vig market probability logged** — `odds_api` only ever read the `Over`
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
