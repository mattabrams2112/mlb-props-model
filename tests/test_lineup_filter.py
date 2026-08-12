"""
Guards the Game View lineup filter.

`*_batters` carries every player ID the MLB API listed for a side; only
`*_batter_codes` holds real batting-order positions. Game View iterated the
former, so IDs missing from the codes map fell through
`batter_codes.get(int(pid), (idx + 1) * 100)` to a synthetic code that satisfies
`ocode % 100 == 0` — meaning they were classified as STARTERS and then shown as
"Not enough data" when projection failed. Every game on 2026-08-11 was affected:
10 vs 9 IDs pre-game (the starting pitcher), up to 17 vs 11 once relievers and
pinch-hitters had appeared. worker.py was already correct; it defaults to 0,
which fails its own `ocode > 0` test.

Two layers here, deliberately separate:

  * REGRESSION — a pinned, sanitized copy of the 2026-08-11 payload that exposed
    the bug. It asserts the exact historical extras are removed. Offline: the
    fixture is a file, because the live API can revise historical payloads and a
    regression baseline that moves is not a baseline.
  * INVARIANTS — date-independent properties that must hold for ANY payload.
    These deliberately do NOT assert that anything was removed: a future payload
    containing only validly coded hitters is correct, and the test must not
    depend on MLB continuing to send noise forever.

KNOWN LIMITATION — small testability cleanup, not yet done:
The behavioural tests run a COPY of the predicate (`_apply_filter`), and
`test_game_view_still_filters_batter_ids` only proves the shipped comprehension
references `batter_codes` — not that it applies the correct membership test.
A comprehension filtering on the wrong condition would still satisfy it.

The strongest structure is to extract the filter into a small shared function,
e.g. `filter_batting_order_players(batter_ids, batter_codes)`, have
`render_lineup` call it, and test that production function directly. The copy
could then not drift from the shipped implementation, and the AST test would
become unnecessary and should be deleted. Deferred deliberately: it is a source
change, and this commit is test-only.

Run: python -m pytest tests/test_lineup_filter.py -q
"""
import ast
import json
import os
import socket
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIXTURE   = os.path.join(REPO, 'tests', 'fixtures', 'lineups_2026-08-11.json')
GAME_VIEW = os.path.join(REPO, 'pages', '2_\U0001f3af_Game_View.py')

# Extra IDs the filter removed from the 2026-08-11 payload, per side. Evidence
# about that specific payload — NOT an invariant. See module docstring.
REGRESSION_BASELINE = {
    ('CLE@DET', 'away'): 3, ('CLE@DET', 'home'): 6,
    ('PIT@MIA', 'away'): 2, ('PIT@MIA', 'home'): 3,
    ('CHC@WSH', 'away'): 5, ('CHC@WSH', 'home'): 4,
    ('SEA@NYY', 'away'): 5, ('SEA@NYY', 'home'): 4,
    ('BOS@TOR', 'away'): 3, ('BOS@TOR', 'home'): 5,
    ('NYM@ATL', 'away'): 3, ('NYM@ATL', 'home'): 4,
    ('BAL@MIN', 'away'): 4, ('BAL@MIN', 'home'): 4,
    ('CIN@CWS', 'away'): 6, ('CIN@CWS', 'home'): 4,
    ('PHI@STL', 'away'): 3, ('PHI@STL', 'home'): 5,
    ('TEX@LAA', 'away'): 2, ('TEX@LAA', 'home'): 5,
    ('COL@ARI', 'away'): 4, ('COL@ARI', 'home'): 4,
    ('TB@OAK',  'away'): 1, ('TB@OAK',  'home'): 4,
    ('MIL@SD',  'away'): 2, ('MIL@SD',  'home'): 3,
    ('HOU@SF',  'away'): 4, ('HOU@SF',  'home'): 4,
    ('KC@LAD',  'away'): 4, ('KC@LAD',  'home'): 6,
}
EXPECTED_SIDES  = 30
EXPECTED_EXTRAS = 116


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any socket use fails the test. This suite must never reach the network."""
    def _blocked(*a, **k):
        raise AssertionError(
            'network access attempted — this regression must run offline against '
            'the pinned fixture, never against the live MLB API')
    monkeypatch.setattr(socket, 'socket', _blocked)
    monkeypatch.setattr(socket, 'create_connection', _blocked)


def _sides():
    """(label, side, batter_ids, batter_codes) for every coded side in the fixture."""
    with open(FIXTURE, encoding='utf-8') as fh:
        doc = json.load(fh)
    out = []
    for g in doc['games']:
        label = f"{g['away_team']}@{g['home_team']}"
        for side in ('away', 'home'):
            codes = g.get(f'{side}_batter_codes') or {}
            if not codes:
                continue
            # JSON forces string keys; the real codes map is keyed by int.
            out.append((label, side, g.get(f'{side}_batters') or [],
                        {int(k): v for k, v in codes.items()}))
    return out


def _apply_filter(batter_ids, batter_codes):
    """The predicate shipped in render_lineup. Kept in sync by the AST test below."""
    if batter_codes:
        return [pid for pid in batter_ids if int(pid) in batter_codes]
    return list(batter_ids)


def _split(kept, codes):
    """Starters have ocode % 100 == 0; anything else is a coded substitution."""
    starters = [p for p in kept if codes[int(p)] % 100 == 0]
    subs     = [p for p in kept if codes[int(p)] % 100 != 0]
    return starters, subs


# ── Regression: the pinned 2026-08-11 payload ────────────────────────────────

def test_fixture_covers_every_side():
    assert len(_sides()) == EXPECTED_SIDES


def test_fixture_removes_the_recorded_extras():
    total = 0
    for label, side, ids, codes in _sides():
        removed = len(ids) - len(_apply_filter(ids, codes))
        expected = REGRESSION_BASELINE.get((label, side))
        assert expected is not None, f'no baseline recorded for {label} {side}'
        assert removed == expected, (
            f'{label} {side}: filter removed {removed}, fixture recorded {expected}')
        total += removed
    assert total == EXPECTED_EXTRAS


# ── Invariants: must hold for any payload, noisy or clean ────────────────────

def test_exactly_nine_starters_per_side():
    for label, side, ids, codes in _sides():
        starters, _ = _split(_apply_filter(ids, codes), codes)
        assert len(starters) == 9, f'{label} {side}: {len(starters)} starters'


def test_starter_spots_are_one_through_nine_exactly_once():
    for label, side, ids, codes in _sides():
        starters, _ = _split(_apply_filter(ids, codes), codes)
        spots = sorted(codes[int(p)] // 100 for p in starters)
        assert spots == list(range(1, 10)), f'{label} {side}: spots {spots}'


def test_no_duplicate_ids():
    for label, side, ids, codes in _sides():
        kept = _apply_filter(ids, codes)
        assert len(set(kept)) == len(kept), f'{label} {side}: duplicate player ids'
        starters, _ = _split(kept, codes)
        assert len(set(starters)) == len(starters), f'{label} {side}: duplicate starters'


def test_substitutions_are_excluded_from_starter_totals():
    """Subs are legitimately coded (601 = sub for spot 6) and must not count."""
    saw_a_sub = False
    for label, side, ids, codes in _sides():
        kept = _apply_filter(ids, codes)
        starters, subs = _split(kept, codes)
        saw_a_sub |= bool(subs)
        assert set(starters).isdisjoint(subs), f'{label} {side}: sub counted as starter'
        assert len(starters) + len(subs) == len(kept), f'{label} {side}: lost a row'
        for p in subs:
            assert codes[int(p)] % 100 != 0, f'{label} {side}: {p} misclassified'
    assert saw_a_sub, 'fixture should be a completed slate containing substitutions'


def test_pitcher_is_not_rendered():
    """A pitcher has no batting-order code, so he must never survive the filter."""
    for label, side, ids, codes in _sides():
        for pid in _apply_filter(ids, codes):
            assert int(pid) in codes, f'{label} {side}: {pid} kept without a code'


# ── The shipped source must still apply the filter ───────────────────────────

def test_game_view_still_filters_batter_ids():
    """
    The behavioural tests above run a copy of the predicate. This asserts the
    real one is still there, so the copy cannot pass while Game View regresses.
    """
    with open(GAME_VIEW, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == 'render_lineup'), None)
    assert fn is not None, 'render_lineup no longer exists in Game View'

    found = False
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, 'id', '') == 'batter_ids' for t in node.targets)
                and isinstance(node.value, ast.ListComp)):
            src = ast.unparse(node.value)
            if 'batter_codes' in src:
                found = True
    assert found, (
        'render_lineup no longer filters batter_ids against batter_codes. '
        'Without it, IDs absent from the codes map fall through to '
        '(idx + 1) * 100, satisfy ocode %% 100 == 0, and are rendered as '
        'starters — which is how the starting pitcher showed "Not enough data".')
