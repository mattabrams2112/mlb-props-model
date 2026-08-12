"""
Guards the worker's per-side outcome funnel.

Until 2026-08-12 `_run_prediction` ended with a bare `except RuntimeError:
return None  # insufficient history for this batter`. Both LogFetchError and
InsufficientData subclass RuntimeError, so an MLB API outage was silently
recorded as "this batter lacks history" — the same conflation removed from Game
View in e51311f, which never reached the worker because that commit was scoped
to the Game View path. Nothing was logged either way, so a slate that scored
four batters instead of nine looked identical to one where five were genuinely
ineligible.

These tests pin two things:
  * each exception type lands in its own bucket, in the right order (the two
    subclasses must be caught before the bare RuntimeError, or they regress
    into it silently);
  * the counters reconcile, so an outcome cannot escape classification.

Run: python -m pytest tests/test_worker_funnel.py -q
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import worker                                    # noqa: E402
from game_log_fetcher import LogFetchError       # noqa: E402
from projection import InsufficientData          # noqa: E402

ARGS = (12345, 999, True, 'LAD', 72.0, 5.0, 0, '2026-08-12')


@pytest.fixture(autouse=True)
def _clear_memo():
    """_PRED_CACHE would return a hit and skip the attempt entirely."""
    worker._PRED_CACHE.clear()
    yield
    worker._PRED_CACHE.clear()


def _run(monkeypatch, outcome):
    """Drive _run_prediction with a stubbed pipeline; return (result, funnel)."""
    def fake(*a, **k):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(worker, 'shared_run_prediction', fake)
    funnel = worker.Funnel()
    result = worker._run_prediction(*ARGS, funnel=funnel)
    return result, funnel


# ── each exception reaches its own bucket ────────────────────────────────────

def test_upstream_failure_bucket(monkeypatch):
    result, f = _run(monkeypatch, LogFetchError('simulated timeout'))
    assert result is None
    assert f['upstream_failure'] == 1
    assert f['insufficient_data'] == 0, 'API failure mislabeled as ineligibility'
    assert f['unexpected_failure'] == 0
    assert f['projected'] == 0


def test_insufficient_data_bucket(monkeypatch):
    result, f = _run(monkeypatch, InsufficientData(31, usable=16))
    assert result is None
    assert f['insufficient_data'] == 1
    assert f['upstream_failure'] == 0
    assert f['unexpected_failure'] == 0


def test_other_runtime_error_is_not_mislabeled(monkeypatch):
    """A plain RuntimeError is an unexpected failure, not an eligibility one."""
    result, f = _run(monkeypatch, RuntimeError('something else broke'))
    assert result is None
    assert f['unexpected_failure'] == 1
    assert f['insufficient_data'] == 0
    assert f['upstream_failure'] == 0


def test_non_runtime_exception_is_unexpected(monkeypatch):
    result, f = _run(monkeypatch, ValueError('bad value'))
    assert result is None
    assert f['unexpected_failure'] == 1


def test_success_counts_as_projected(monkeypatch):
    result, f = _run(monkeypatch, {'proj': 2.4})
    assert result == {'proj': 2.4}
    assert f['projected'] == 1
    assert f['insufficient_data'] == f['upstream_failure'] == f['unexpected_failure'] == 0


# ── the subclasses must be caught before the bare RuntimeError ───────────────

@pytest.mark.parametrize('exc,bucket', [
    (LogFetchError('x'),            'upstream_failure'),
    (InsufficientData(10),          'insufficient_data'),
    (RuntimeError('x'),             'unexpected_failure'),
])
def test_runtime_error_subclasses_do_not_collapse(monkeypatch, exc, bucket):
    """Reordering the handlers would silently funnel these into one bucket."""
    assert isinstance(exc, RuntimeError), 'test premise: all are RuntimeErrors'
    _, f = _run(monkeypatch, exc)
    assert f[bucket] == 1, f'{type(exc).__name__} did not reach {bucket}'
    others = set(worker.Funnel.FIELDS) - {bucket, 'attempted', 'expected',
                                          'cached', 'persisted', 'projected'}
    for o in others:
        assert f[o] == 0, f'{type(exc).__name__} also incremented {o}'


# ── totals reconcile ─────────────────────────────────────────────────────────

def test_every_outcome_increments_attempted(monkeypatch):
    for outcome in (LogFetchError('x'), InsufficientData(10),
                    RuntimeError('x'), ValueError('x'), {'proj': 1.0}):
        worker._PRED_CACHE.clear()
        _, f = _run(monkeypatch, outcome)
        assert f['attempted'] == 1, f'{outcome!r} did not count as an attempt'
        assert f.outcomes_reconcile(), f'{outcome!r} escaped outcome classification'


def test_coverage_catches_a_vanished_starter():
    """
    The bug this bucket exists for: a post-start run reported expected=9,
    cached=8, attempted=0 and still claimed OK. The ninth batter was skipped
    with no bucket, so the shortfall was invisible.
    """
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump('cached', 8)             # ninth batter unaccounted for
    assert f.outcomes_reconcile(), 'no attempts were made, so outcomes are fine'
    assert not f.coverage_reconciles(), 'a missing starter must not read as OK'
    assert not f.reconciles()
    assert 'coverage=MISMATCH' in f.line('KC (away)')


def test_coverage_reconciles_when_skips_carry_a_reason():
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump('cached', 8)
    f.bump_skip('game_started')     # accounted for, with a declared reason
    assert f.coverage_reconciles()
    assert f.reconciles()
    line = f.line('KC (away)')
    assert 'coverage=OK' in line
    assert 'skip_game_started=1' in line


def test_skip_without_a_reason_fails_coverage():
    """`skipped` must never be a generic bucket that hides missing work."""
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump('cached', 8)
    f.bump('skipped', 1)            # incremented directly, no reason attached
    assert not f.skips_explained()
    assert not f.coverage_reconciles(), 'an unattributed skip must not read as OK'
    assert 'coverage=MISMATCH' in f.line('KC (away)')


def test_unknown_skip_always_fails_coverage():
    """Even fully counted, an 'unknown' skip is an unexplained omission."""
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump('cached', 8)
    f.bump_skip('unknown')
    assert sum(f.skip(r) for r in worker.Funnel.SKIP_REASONS) == f['skipped']
    assert not f.skips_explained(), 'unknown must never count as explained'
    assert not f.coverage_reconciles()
    assert 'skip_unknown=1' in f.line('KC (away)')


def test_undeclared_reason_is_recorded_as_unknown():
    f = worker.Funnel()
    f.bump_skip('some_reason_nobody_declared')
    assert f.skip('unknown') == 1
    assert f['skipped'] == 1
    assert not f.skips_explained()


@pytest.mark.parametrize('reason', worker.Funnel.SKIP_REASONS)
def test_every_declared_reason_is_countable(reason):
    f = worker.Funnel()
    f.bump_skip(reason)
    assert f.skip(reason) == 1
    assert f['skipped'] == 1


# ── persistence: projected == persisted + persistence_failure ────────────────

def test_persistence_reconciles_when_all_projected_persist():
    f = worker.Funnel()
    f.bump('projected', 4)
    f.bump('persisted', 4)
    assert f.persistence_reconciles()
    assert 'persistence=OK' in f.line('LAD (home)')


def test_projected_but_not_persisted_fails_without_a_recorded_failure():
    """A batter that projects and then vanishes before the DB must show up."""
    f = worker.Funnel()
    f.bump('projected', 4)
    f.bump('persisted', 3)          # one lost, nothing recorded
    assert not f.persistence_reconciles()
    assert 'persistence=MISMATCH' in f.line('LAD (home)')


def test_recorded_persistence_failure_reconciles():
    f = worker.Funnel()
    f.bump('projected', 4)
    f.bump('persisted', 3)
    f.bump('persistence_failure', 1)
    assert f.persistence_reconciles()
    assert 'persistence=OK' in f.line('LAD (home)')


def test_cached_batters_do_not_require_persistence():
    """cached=8 persisted=0 is normal: they were persisted on an earlier run."""
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump('cached', 8)
    f.bump_skip('game_started')
    assert f['projected'] == 0 and f['persisted'] == 0
    assert f.persistence_reconciles()
    assert f.reconciles()


def test_both_invariants_reported_independently():
    """A run can fail one and pass the other; the line must distinguish them."""
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump('attempted', 9)
    f.bump('projected', 7)          # two outcomes lost -> outcomes MISMATCH
    line = f.line('LAD (home)')
    assert 'outcomes=MISMATCH' in line
    assert 'coverage=OK' in line


def test_mixed_slate_reconciles():
    """attempted == projected + insufficient + upstream + unexpected."""
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump('attempted', 7)
    f.bump('projected', 4)
    f.bump('insufficient_data', 2)
    f.bump('upstream_failure', 1)
    f.bump('cached', 2)
    f.bump('persisted', 4)
    assert f.reconciles()
    assert 'reconcile=OK' in f.line('LAD (home)')


def test_reconcile_detects_an_unclassified_outcome():
    """An attempt with no recorded outcome must show as MISMATCH."""
    f = worker.Funnel()
    f.bump('attempted', 5)
    f.bump('projected', 3)          # two outcomes vanished
    assert not f.reconciles()
    assert 'reconcile=MISMATCH' in f.line('LAD (home)')


def test_line_reports_every_field():
    f = worker.Funnel()
    line = f.line('SD (away)')
    for field in worker.Funnel.FIELDS:
        assert f'{field}=' in line, f'{field} missing from funnel summary'
    assert 'SD (away)' in line
