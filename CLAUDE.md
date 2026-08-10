# MLB Props Model — working notes

Streamlit app + Railway cron that rate MLB batters on H+R+RBI (HRR) props.
`MODEL_DECISIONS.md` is the running log of *why* the model is the way it is —
read it before changing rating or projection logic, and add to it after.

## Architecture

Two things run on Railway, both built from this repo:

| service | what it runs | when |
|---|---|---|
| `web` | `streamlit run app.py` | always |
| `mlb-props-model` | `python worker.py` | cron, every 30 min, 13-23 + 0-4 UTC |

The cron service reads **`railway.worker.toml`**, not `railway.toml` — that is
what gives it a different start command. Cron config is NOT in `railway.toml`;
the `[[cronJobs]]` blocks that used to be there were never valid Railway schema
and were silently ignored for months while the cron booted a web server instead.

Both paths write the same Postgres tables: `full_play_log`, `ratings_cache`,
`game_predictions`.

## The rule that matters most

**Scoring logic lives in exactly two shared modules. Never add a third copy.**

- `projection.py` — the whole projection pipeline (XGBoost + LightGBM ensemble,
  ceiling, pitcher multiplier, lineup context, recency weighting, power/contact
  tuning, feature building).
- `rating_inputs.py` — `score_batter()`, the single place that assembles
  `compute_rating`'s ~80 arguments.

`worker.py` and `pages/2_🎯_Game_View.py` are thin callers. If you need
different behaviour in one of them, put it behind an argument in the shared
module where the difference is visible.

**Why this is a hard rule:** every parameter in `rating.py` has a default, so a
caller that forgets one does not error — it scores that component against a
plausible-looking constant. When these existed twice, three separate features
silently did nothing, one of them for 69 days. See MODEL_DECISIONS.md 2026-08-10.

`tests/test_rating_parity.py` enforces this. A `pre-push` hook runs it.

## Before you push

Any push rebuilds and restarts **both** services (`watchPatterns` is empty), so
a broken push is a live outage.

```
python -m pytest tests/ -q
```

The `.git/hooks/pre-push` hook does this automatically plus a compile check.
It is local-only and not in the repo — reinstall it on a fresh clone.

Static tests are a backstop, not proof. They passed on a version that raised
`TypeError` on every batter, because a `**splat` hides its keys from analysis.
**For any change to scoring, run a real game**:

```python
import os; os.environ['DATABASE_URL']=''          # never write to prod
import worker
from lineup_fetcher import get_todays_lineups
g = [x for x in get_todays_lineups('YYYY-MM-DD') if x.get('home_batter_codes')][0]
worker.process_game(g, 'YYYY-MM-DD')
```

Expect ~100-230s per game: it trains two models per batter and pulls real
pitcher history. That is normal, not a hang.

## Traps that have already bitten

- **`requirements.txt` upper bounds are deliberate.** pandas and numpy were
  unbounded; a rebuild pulled pandas 3.0, which turned a tolerated
  `Timestamp`/`date` comparison into a `TypeError` and silently zeroed every
  projection. Raise the bounds only after running a real game.
- **Ratings freeze on first write** (`ratings_cache.save_rating` refuses to
  overwrite) and Game View honours the frozen value without recomputing. Model
  changes do not affect already-rated players — clear the cache to re-rate.
- **`DATA_DIR` has no volume mounted.** Disk caches are wiped every deploy.
  Postgres persists; local CSVs do not.
- **Play-log rows key on date + player + vs_pitcher.** A probable-pitcher change
  therefore creates a duplicate row instead of updating (~0.93% of rows, all
  double-counted in win rates). The real fix is keying on `game_pk`; not done.
- **Two different active players can share a name** (both Max Muncys). Anything
  keyed on player name needs a second discriminator.

## Betting logic

`bet_config.py` is the single source for which ratings count as tracked bets,
and it is **date-gated** so historical records never move. Do not change
thresholds without adding a MODEL_DECISIONS.md entry explaining the evidence.

## Unvalidated — treat with suspicion

- **Park history** (`park_splits.py`) went live 2026-08-10 and has never been
  tested against outcomes. Gated at `MIN_PARK_AB = 50` because `rating.py`'s own
  `park_ab >= 20` guard scores noise. Mostly only a batter's home park clears it.
- **Calibration** (`calibration.py`) is wired in but its factors are known-bad.
  Fitting is not the blocker; the blocker is that actual HRR is nearly flat
  across every projection level, so an honest calibration collapses toward a
  constant and empties the bet band. It must ship together with a re-derived
  rating threshold, never alone.
- **`umpire_tendency`** is fetched, passed into `compute_rating`, and used by no
  component. Dead parameter.
