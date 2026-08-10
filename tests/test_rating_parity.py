"""
Guards against the failure that hid two features for 69 days.

`compute_rating` gives every one of its ~83 parameters a default, so a caller
that forgets one does not error — it silently scores that component against a
plausible constant. When the projection pipeline existed twice (worker.py and
Game View), each hand-built its own ~80-argument call, and they drifted:

  * park splits and pitch count were wired into worker.py only (commit a561e51,
    2026-06-02). Game View defaulted park_ab to 0, which fails rating.py's
    `if park_ab >= 20` guard, so the whole park-history block scored zero on the
    page that actually generated ratings. Nothing errored for 69 days.
  * batter_obp / batter_sb_rate / batter_rest_days were passed by Game View and
    not the worker, so the two scored Form & Hit Rate and Batter Rest
    differently for the same player.

Both now go through rating_inputs.score_batter. These tests fail if anyone
reintroduces a second call site, or adds a rating parameter that score_batter
never passes.

Run: python -m pytest tests/test_rating_parity.py -q
"""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SHARED   = os.path.join(REPO, 'rating_inputs.py')
CALLERS  = [os.path.join(REPO, 'worker.py'),
            os.path.join(REPO, 'pages', '2_\U0001f3af_Game_View.py')]


def _compute_rating_calls(path):
    """Every compute_rating(...) call in a file, as {param: source expression}."""
    with open(path, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'compute_rating':
            calls.append({(kw.arg or '**splat'): ast.unparse(kw.value)
                          for kw in node.keywords})
    return calls


def test_only_one_place_builds_the_rating_call():
    """No caller may hand-build compute_rating; they must go through score_batter."""
    offenders = [p for p in CALLERS if _compute_rating_calls(p)]
    assert not offenders, (
        'These files build their own compute_rating call again: '
        + ', '.join(os.path.basename(p) for p in offenders)
        + '. Add the input to rating_inputs.score_batter instead — two call '
          'sites is exactly how park splits and pitch count went unnoticed.')


def test_shared_builder_passes_every_rating_parameter():
    """score_batter must supply every compute_rating parameter, not lean on defaults."""
    import inspect
    from rating import compute_rating

    accepted = {p for p in inspect.signature(compute_rating).parameters}
    calls = _compute_rating_calls(SHARED)
    assert len(calls) == 1, 'rating_inputs.py should hold exactly one compute_rating call'
    passed = set(calls[0])

    # No **splat. get_batter_park_splits also returns park_ops, which
    # compute_rating rejects — splatting it raised TypeError on every batter and
    # this test could not see it, because a splat hides its keys from static
    # analysis. Keep every argument named so both directions stay checkable.
    assert '**splat' not in passed, (
        'score_batter must pass arguments by name, not **splat: a splat can '
        'smuggle in a keyword compute_rating does not accept (park_ops did '
        'exactly that) and no static check can catch it.')

    missing = sorted(accepted - passed - {'self'})
    assert not missing, (
        f'score_batter never passes {missing}. Each would silently fall back to '
        'its default and score as a constant. Pass it explicitly, or delete the '
        'parameter from rating.py if it is genuinely unused.')

    unknown = sorted(passed - accepted)
    assert not unknown, (
        f'score_batter passes {unknown}, which compute_rating does not accept — '
        'this is a TypeError at runtime for every batter.')


def test_callers_import_the_shared_builder():
    for path in CALLERS:
        with open(path, encoding='utf-8') as fh:
            src = fh.read()
        assert 'score_batter' in src, f'{os.path.basename(path)} no longer uses score_batter'
