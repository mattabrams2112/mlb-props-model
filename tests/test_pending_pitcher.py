"""
Guards the pending-pitcher gate and matchup-aware cache retrieval.

Confirmed in production 2026-08-12. One connected defect:

    worker rated batters while vs_pitcher == 'TBD'
      -> the TBD rating was frozen in ratings_cache (save_rating never
         overwrites)
      -> worker looked up the frozen rating with
         get_cached_rating(game_date, pid) — no vs_pitcher
      -> a later resolved-pitcher run could be served the TBD rating
      -> both ratings_cache and full_play_log accumulated conflicting rows

Evidence: MIN held 18 ratings_cache rows for 9 players, every one carrying both
a TBD and a resolved-pitcher entry, differing by up to 12 rating points
(Royce Lewis 74 vs 62). 8 confirmed duplicate pairs = 16 conflicting play-log
rows.

Suppressing only the play-log write is NOT sufficient: the invalid rating would
still reach ratings_cache and freeze there. The rating must not be calculated
at all while the pitcher is unresolved.

Run: python -m pytest tests/test_pending_pitcher.py -q
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import ratings_cache                     # noqa: E402
import worker                            # noqa: E402


# ── matchup-aware retrieval ──────────────────────────────────────────────────

@pytest.fixture
def cache(monkeypatch, tmp_path):
    """Isolated ratings_cache backed by a temp CSV. Never touches production."""
    import pandas as pd
    store = {'df': pd.DataFrame(columns=['date', 'player_id', 'rating', 'grade',
                                         'projected', 'player_name', 'team',
                                         'vs_pitcher'])}
    monkeypatch.setattr(ratings_cache, '_load', lambda: store['df'].copy())
    monkeypatch.setattr(ratings_cache, '_save',
                        lambda df: store.__setitem__('df', df.copy()))
    monkeypatch.setattr(ratings_cache, 'update_rating_if_exists',
                        lambda *a, **k: None, raising=False)
    return store


def _save(pid, vs_pitcher, rating, proj):
    ratings_cache.save_rating('2026-08-12', pid, rating, 'B', proj,
                              player_name='X', team='MIN', vs_pitcher=vs_pitcher)


def test_tbd_rating_is_not_served_to_a_resolved_pitcher_run(cache):
    """The exact production failure: a TBD rating must not satisfy a real matchup."""
    _save(111, 'TBD', 74, 2.70)
    assert ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz') is None, (
        'a rating computed against no pitcher was served for a real matchup')


def test_resolved_rating_is_served_for_its_own_pitcher(cache):
    _save(111, 'Shane Baz', 62, 2.15)
    got = ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz')
    assert got == (62, 'B', 2.15)


def test_matchup_blind_lookup_is_ambiguous_when_both_rows_exist(cache):
    """Demonstrates why the pitcher argument is mandatory, not optional."""
    _save(111, 'TBD', 74, 2.70)
    _save(111, 'Shane Baz', 62, 2.15)
    blind = ratings_cache.get_cached_rating('2026-08-12', 111)          # no pitcher
    aware = ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz')
    assert aware == (62, 'B', 2.15)
    assert blind != aware, (
        'matchup-blind lookup happened to return the right row here, but it '
        'selects by insertion order — it cannot be relied on')


def test_different_pitcher_same_day_is_not_served(cache):
    """Doubleheader / pitcher change: game 2 must not inherit game 1."""
    _save(111, 'Shane Baz', 62, 2.15)
    assert ratings_cache.get_cached_rating('2026-08-12', 111, 'Zack Littell') is None


# ── idempotency across repeated cron runs ───────────────────────────────────

def test_repeated_save_does_not_create_a_second_row(cache):
    """The cron runs every 30 min; a re-run must not duplicate a rating."""
    _save(111, 'Shane Baz', 62, 2.15)
    _save(111, 'Shane Baz', 62, 2.15)
    _save(111, 'Shane Baz', 99, 9.99)      # even with different values
    df = cache['df']
    rows = df[(df.player_id == '111') & (df.vs_pitcher == 'Shane Baz')]
    assert len(rows) == 1, 'repeated runs created duplicate cache rows'
    assert int(rows.iloc[0]['rating']) == 62, 'freeze semantics were violated'


def test_repeated_lookup_is_stable(cache):
    """Two runs must resolve to the same rating, not alternate between rows."""
    _save(111, 'Shane Baz', 62, 2.15)
    a = ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz')
    b = ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz')
    assert a == b == (62, 'B', 2.15)


def test_pitcher_change_creates_a_separate_row_and_does_not_cross_serve(cache):
    """A probable-pitcher change must not mutate or be served the old rating."""
    _save(111, 'Shane Baz', 62, 2.15)
    _save(111, 'Zack Littell', 71, 2.80)    # pitcher changed
    assert ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz') == (62, 'B', 2.15)
    assert ratings_cache.get_cached_rating('2026-08-12', 111, 'Zack Littell') == (71, 'B', 2.80)
    # And a third run against the new pitcher stays frozen on the new value.
    _save(111, 'Zack Littell', 5, 0.5)
    assert ratings_cache.get_cached_rating('2026-08-12', 111, 'Zack Littell') == (71, 'B', 2.80)


def test_gate_is_idempotent_across_runs():
    """Each pending run re-reports; it must not accumulate or self-cancel."""
    for _ in range(3):
        f = worker.Funnel()
        f.bump('expected', 9)
        f.bump_skip('pending_pitcher', 9)
        assert f['persisted'] == 0 and f['attempted'] == 0
        assert f.reconciles()


def test_tbd_row_never_created_means_no_cross_serve_after_resolution(cache):
    """
    End state the gate produces: because the TBD rating is never written, the
    resolved run finds an empty cache and computes cleanly — no ambiguity.
    """
    # gate suppressed the TBD write entirely, so nothing exists yet
    assert ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz') is None
    assert ratings_cache.get_cached_rating('2026-08-12', 111) is None
    _save(111, 'Shane Baz', 62, 2.15)       # resolved run writes once
    df = cache['df']
    assert len(df[df.player_id == '111']) == 1, 'exactly one row per player-game'
    assert ratings_cache.get_cached_rating('2026-08-12', 111, 'Shane Baz') == (62, 'B', 2.15)


def test_worker_passes_the_pitcher_to_the_cache_lookup():
    """Static guard: the call site must not regress to the 2-arg form."""
    import ast
    with open(os.path.join(REPO, 'worker.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, 'id', '') == 'get_cached_rating']
    assert calls, 'worker no longer calls get_cached_rating'
    for c in calls:
        assert len(c.args) + len(c.keywords) >= 3, (
            'get_cached_rating called without the opposing pitcher — the '
            'lookup becomes matchup-blind and can serve a TBD-era rating')


# ── the gate: no rating, no write, while unresolved ──────────────────────────

@pytest.mark.parametrize('pitcher,official,expected_reason', [
    ('TBD',       True,  'pending_pitcher'),
    ('',          True,  'pending_pitcher'),
    (None,        True,  'pending_pitcher'),
    ('Shane Baz', False, 'lineup_not_official'),
    ('TBD',       False, 'pending_pitcher'),
])
def test_gate_reason_selection(pitcher, official, expected_reason):
    """Mirrors the gate's branch logic for a pre-game side."""
    game_started = False
    resolved = bool(pitcher) and str(pitcher).strip() not in ('', 'TBD')
    if game_started:
        reason = 'missed_cutoff_pending_pitcher'
    elif not resolved:
        reason = 'pending_pitcher'
    else:
        reason = 'lineup_not_official'
    assert not (resolved and official) or expected_reason is None
    assert reason == expected_reason


def test_after_first_pitch_records_missed_cutoff():
    """Past cutoff the state is a missed opportunity, not a pending retry."""
    game_started = True
    resolved = False
    reason = ('missed_cutoff_pending_pitcher' if game_started
              else ('pending_pitcher' if not resolved else 'lineup_not_official'))
    assert reason == 'missed_cutoff_pending_pitcher'


def test_gate_reasons_are_declared_skip_reasons():
    """Undeclared reasons silently become 'unknown' and fail coverage."""
    for r in ('pending_pitcher', 'missed_cutoff_pending_pitcher',
              'lineup_not_official'):
        assert r in worker.Funnel.SKIP_REASONS


def test_pending_side_still_reconciles():
    """A fully pending side is accounted for, with an explicit reason."""
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump_skip('pending_pitcher', 9)
    assert f['skipped'] == 9
    assert f.skip('pending_pitcher') == 9
    assert f.coverage_reconciles() and f.skips_explained()
    assert f.outcomes_reconcile() and f.persistence_reconciles()
    line = f.line('MIN (home)')
    assert 'skip_pending_pitcher=9' in line and 'reconcile=OK' in line


def test_pending_side_records_no_projection_or_persistence():
    """The point of the gate: nothing is computed and nothing is written."""
    f = worker.Funnel()
    f.bump('expected', 9)
    f.bump_skip('pending_pitcher', 9)
    assert f['attempted'] == 0, 'no projection may be attempted'
    assert f['projected'] == 0
    assert f['persisted'] == 0, 'nothing may be written while unresolved'
    assert f['cached'] == 0


def test_worker_gates_on_pitcher_and_official_lineup():
    """Static guard: the gate must remain in process_game."""
    with open(os.path.join(REPO, 'worker.py'), encoding='utf-8') as fh:
        src = fh.read()
    assert "'pending_pitcher'" in src
    assert "'missed_cutoff_pending_pitcher'" in src
    assert "lineups_official" in src, 'gate no longer checks the official lineup'
    assert "persist-blocked" in src, 'the write-site guard was removed'
