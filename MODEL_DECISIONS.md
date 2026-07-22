# Model Decisions Log

A running record of *why* the rating/projection/betting logic is the way it is —
so decisions don't get re-litigated from scratch. Newest first. Dates are ET.

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
