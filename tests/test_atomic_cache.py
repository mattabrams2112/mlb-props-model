"""
Guards the CSV cache concurrency fix.

statcast_features, bvp_stats and bullpen_data each returned their module-level
`_MEM_CACHE` by reference, so callers mutated the same dict the saver was
iterating:

    cache[key] = result                # thread A
    for k, v in cache.items()          # thread B, inside _save_cache

Under the worker's 8-thread pool that raises `RuntimeError: dictionary changed
size during iteration`. Seven batters were dropped in the 2026-08-12 18:32
production run, and the same error appears at 17:02, 17:31 and 18:01. Before the
funnel existed it was swallowed as "insufficient history".

Snapshotting alone is insufficient — concurrent `to_csv` writers truncate each
other, and a mid-write read used to return `{}` and silently discard the whole
cache. These tests cover all three: no iteration error, no lost entries, no
partial file, and existing data preserved when a write fails.

Run: python -m pytest tests/test_atomic_cache.py -q
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import atomic_cache  # noqa: E402


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / 'cache_test.csv')


# ── negative control: the old pattern really does fail ──────────────────────

def test_old_shared_dict_pattern_raises_under_concurrency():
    """Without this failing, the rest of the suite proves nothing."""
    shared = {f'seed{i}': {'v': i} for i in range(20000)}
    errors = []
    stop = threading.Event()

    def mutate():
        i = 0
        while not stop.is_set():
            shared[f'new{i}'] = {'v': i}
            i += 1

    def iterate():
        # The exact shape of the old _save_cache. NOT list(d.items()) — that is
        # a single C-level call and effectively atomic under the GIL, which is
        # why a first draft of this control failed to reproduce anything. The
        # production line is a Python-level comprehension with ** unpacking, so
        # it executes bytecode per item and the interpreter can switch threads
        # mid-iteration.
        try:
            for _ in range(200):
                [{'key': k, **v} for k, v in shared.items()]
        except RuntimeError as e:
            errors.append(str(e))

    t = threading.Thread(target=mutate, daemon=True)
    t.start()
    iterate()
    stop.set()
    t.join(timeout=2)
    assert errors, 'expected the unguarded pattern to raise'
    assert 'changed size during iteration' in errors[0]


# ── the fix: concurrent writers ─────────────────────────────────────────────

def test_concurrent_writers_lose_no_entries(cache_path):
    """Every key written by every thread must survive."""
    mem = {}
    n_threads, per_thread = 8, 40

    def worker(t):
        for i in range(per_thread):
            atomic_cache.update_cache(cache_path, 'key', mem,
                                      f't{t}_k{i}', {'v': t * 1000 + i})

    with ThreadPoolExecutor(max_workers=n_threads) as exe:
        list(exe.map(worker, range(n_threads)))

    on_disk = atomic_cache.load_cache(cache_path, 'key')
    assert len(on_disk) == n_threads * per_thread
    for t in range(n_threads):
        for i in range(per_thread):
            assert f't{t}_k{i}' in on_disk, f'lost t{t}_k{i}'
            assert int(on_disk[f't{t}_k{i}']['v']) == t * 1000 + i


def test_concurrent_readers_and_writers_never_see_a_partial_file(cache_path):
    """A reader must always parse a complete CSV, never a half-written one."""
    mem = {f'seed{i}': {'v': i} for i in range(200)}
    atomic_cache.save_cache(cache_path, 'key', mem)
    failures, sizes = [], []
    stop = threading.Event()

    def writer(t):
        for i in range(60):
            if stop.is_set():
                return
            atomic_cache.update_cache(cache_path, 'key', mem, f'w{t}_{i}', {'v': i})

    def reader():
        while not stop.is_set():
            try:
                d = atomic_cache.load_cache(cache_path, 'key')
                sizes.append(len(d))
            except Exception as e:
                failures.append(f'{type(e).__name__}: {e}')

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for r in readers:
        r.start()
    with ThreadPoolExecutor(max_workers=6) as exe:
        list(exe.map(writer, range(6)))
    stop.set()
    for r in readers:
        r.join(timeout=3)

    assert not failures, f'readers saw torn or unparseable files: {failures[:3]}'
    assert sizes, 'readers never managed a read'
    assert min(sizes) >= 200, 'a read observed fewer rows than the seeded baseline'


def test_no_iteration_error_when_mutating_during_save(cache_path):
    """The exact production failure, driven hard against the new path."""
    mem = {f'seed{i}': {'v': i} for i in range(5000)}
    errors = []
    stop = threading.Event()

    def saver():
        try:
            for _ in range(30):
                atomic_cache.save_cache(cache_path, 'key', mem)
        except RuntimeError as e:
            errors.append(str(e))
        finally:
            stop.set()

    def mutator(t):
        i = 0
        while not stop.is_set():
            atomic_cache.update_cache(cache_path, 'key', mem, f'm{t}_{i}', {'v': i})
            i += 1

    threads = [threading.Thread(target=mutator, args=(t,), daemon=True)
               for t in range(4)]
    for t in threads:
        t.start()
    saver()
    for t in threads:
        t.join(timeout=3)
    assert not errors, f'iteration error still occurs: {errors[:2]}'


# ── failure injection ───────────────────────────────────────────────────────

def test_serialisation_failure_preserves_existing_cache(cache_path, monkeypatch):
    good = {'a': {'v': 1}, 'b': {'v': 2}}
    atomic_cache.save_cache(cache_path, 'key', good)

    def boom(*a, **k):
        raise IOError('simulated disk failure during serialisation')

    monkeypatch.setattr(pd.DataFrame, 'to_csv', boom)
    with pytest.raises(IOError):
        atomic_cache.save_cache(cache_path, 'key', {'c': {'v': 3}})

    monkeypatch.undo()
    assert atomic_cache.load_cache(cache_path, 'key') == {
        'a': {'v': 1}, 'b': {'v': 2}}, 'existing cache was destroyed by a failed write'


def test_replacement_failure_preserves_existing_cache(cache_path, monkeypatch):
    good = {'a': {'v': 1}}
    atomic_cache.save_cache(cache_path, 'key', good)

    monkeypatch.setattr(os, 'replace',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('replace failed')))
    with pytest.raises(OSError):
        atomic_cache.save_cache(cache_path, 'key', {'z': {'v': 9}})

    monkeypatch.undo()
    assert atomic_cache.load_cache(cache_path, 'key') == {'a': {'v': 1}}


def test_failed_write_leaves_no_temp_file(cache_path, monkeypatch):
    atomic_cache.save_cache(cache_path, 'key', {'a': {'v': 1}})
    monkeypatch.setattr(os, 'replace',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('nope')))
    with pytest.raises(OSError):
        atomic_cache.save_cache(cache_path, 'key', {'b': {'v': 2}})
    monkeypatch.undo()

    directory = os.path.dirname(cache_path)
    leftovers = [f for f in os.listdir(directory) if f.endswith('.tmp')]
    assert not leftovers, f'abandoned temp files: {leftovers}'


def test_temp_file_is_written_in_the_destination_directory(cache_path, monkeypatch):
    """os.replace is only atomic within one filesystem."""
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        seen['src_dir'] = os.path.dirname(os.path.abspath(src))
        seen['dst_dir'] = os.path.dirname(os.path.abspath(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, 'replace', spy)
    atomic_cache.save_cache(cache_path, 'key', {'a': {'v': 1}})
    assert seen['src_dir'] == seen['dst_dir']


def test_cleanup_removes_stale_temps(cache_path):
    directory = os.path.dirname(cache_path)
    base = os.path.basename(cache_path)
    for n in ('1', '2'):
        open(os.path.join(directory, f'.{base}.{n}.tmp'), 'w').close()
    assert atomic_cache.cleanup_stale_temps(cache_path) == 2
    assert not [f for f in os.listdir(directory) if f.endswith('.tmp')]


# ── missing vs corrupt ──────────────────────────────────────────────────────

def test_missing_file_is_a_cold_start_not_an_error(cache_path):
    assert atomic_cache.load_cache(cache_path, 'key') == {}


def test_corrupt_file_raises_instead_of_silently_emptying(cache_path):
    """The old `except Exception: return {}` turned corruption into total loss."""
    with open(cache_path, 'w', encoding='utf-8') as fh:
        fh.write('key,v\n"unterminated\n\x00\x00binary')
    with pytest.raises(atomic_cache.CacheCorruptionError):
        atomic_cache.load_cache(cache_path, 'key')


def test_same_path_shares_one_lock(tmp_path):
    p = str(tmp_path / 'x.csv')
    assert atomic_cache.lock_for(p) is atomic_cache.lock_for(p)
    assert atomic_cache.lock_for(p) is atomic_cache.lock_for(
        os.path.join(str(tmp_path), '.', 'x.csv'))
    assert atomic_cache.lock_for(p) is not atomic_cache.lock_for(str(tmp_path / 'y.csv'))


def test_lock_registry_is_process_global_across_cache_modules(tmp_path):
    """
    A per-path lock is only correct if ONE registry backs every module. If each
    module owned its own registry, two modules writing the same path would take
    different locks and serialise nothing — the bug would survive the fix while
    every single-module test still passed.
    """
    import bullpen_data
    import bvp_stats
    import pitcher_data
    import statcast_features

    modules = (statcast_features, bvp_stats, bullpen_data, pitcher_data)

    # All four must reference the same module object, hence the same registry.
    assert len({id(m.atomic_cache) for m in modules}) == 1
    assert all(m.atomic_cache is atomic_cache for m in modules)
    assert len({id(m.atomic_cache._LOCKS) for m in modules}) == 1

    # And the same path requested through different modules yields one lock.
    shared_path = str(tmp_path / 'shared.csv')
    locks = [m.atomic_cache.lock_for(shared_path) for m in modules]
    assert len({id(l) for l in locks}) == 1, (
        'identical paths received different locks across modules')


def test_two_modules_writing_one_path_do_not_corrupt_it(tmp_path):
    """Cross-module serialisation, exercised rather than assumed."""
    import bvp_stats
    import statcast_features

    shared = str(tmp_path / 'contended.csv')
    mem = {}
    failures = []

    def write_via(module, tag):
        try:
            for i in range(40):
                with module.atomic_cache.lock_for(shared):
                    mem[f'{tag}{i}'] = {'v': i}
                    module.atomic_cache.save_cache(shared, 'key', mem)
        except Exception as e:
            failures.append(f'{tag}: {type(e).__name__}: {e}')

    threads = [threading.Thread(target=write_via, args=(statcast_features, 'a')),
               threading.Thread(target=write_via, args=(bvp_stats, 'b'))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    assert not failures, failures
    on_disk = atomic_cache.load_cache(shared, 'key')
    assert len(on_disk) == 80, f'lost entries across modules: {len(on_disk)}'


# ── a failed write must not destroy the caller's result ─────────────────────

def test_best_effort_persist_swallows_write_failure(cache_path, monkeypatch):
    """
    A disk failure must not propagate out of get_batter_statcast into the
    worker's `except Exception` and drop the batter. The value is in memory and
    is returned either way; the only cost is a refetch.
    """
    monkeypatch.setattr(os, 'replace',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
    ok = atomic_cache.save_cache_best_effort(cache_path, 'key', {'a': {'v': 1}})
    assert ok is False, 'a failed write must report failure, not raise'


def test_best_effort_persist_reports_success(cache_path):
    assert atomic_cache.save_cache_best_effort(cache_path, 'key', {'a': {'v': 1}}) is True
    assert atomic_cache.load_cache(cache_path, 'key') == {'a': {'v': 1}}


@pytest.mark.parametrize('module', ['statcast_features', 'bvp_stats', 'bullpen_data'])
def test_call_sites_use_best_effort_persist(module):
    """`_put` must not let a write error escape to the projection pipeline."""
    src = open(os.path.join(REPO, f'{module}.py'), encoding='utf-8').read()
    put = src.split('def _put', 1)[1].split('\ndef ', 1)[0]
    assert 'save_cache_best_effort' in put, (
        f'{module}._put persists with the raising variant — a disk hiccup would '
        'surface as unexpected_failure and drop the batter')


# ── failure logs: actionable, but no absolute paths ─────────────────────────

def test_failure_line_names_module_and_basename_but_not_the_directory(tmp_path):
    path = str(tmp_path / 'secret_dir' / 'cache_statcast.csv')
    try:
        raise ValueError('boom')
    except ValueError as inner:
        try:
            raise atomic_cache.CacheCorruptionError(
                f'{path} exists but could not be parsed') from inner
        except atomic_cache.CacheCorruptionError as e:
            line = atomic_cache.describe_failure(
                'cache-read-fail', 'statcast_features', path, e, 'refetch')

    assert 'module=statcast_features' in line
    assert 'file=cache_statcast.csv' in line
    assert 'action=refetch' in line
    # The underlying cause, not the wrapper.
    assert 'error=ValueError' in line
    # No directory anywhere in the line.
    assert 'secret_dir' not in line, 'parent directory leaked into the log'
    assert str(tmp_path) not in line, 'absolute path leaked into the log'
    assert os.sep not in line.split('file=')[1], 'basename contains a separator'


def test_failure_line_without_a_cause_reports_the_exception_itself():
    line = atomic_cache.describe_failure(
        'cache-write-fail', 'bvp_stats', '/a/b/cache_bvp.csv',
        OSError('disk full'), 'continue-in-memory')
    assert 'module=bvp_stats file=cache_bvp.csv error=OSError' in line
    assert 'action=continue-in-memory' in line
    assert '/a/b' not in line


@pytest.mark.parametrize('module,cache_const', [
    ('statcast_features', 'CACHE_FILE'),
    ('bvp_stats', 'CACHE_FILE'),
    ('bullpen_data', 'CACHE_FILE'),
])
def test_modules_log_structured_failures_not_raw_exceptions(module, cache_const):
    src = open(os.path.join(REPO, f'{module}.py'), encoding='utf-8').read()
    assert 'describe_failure' in src, f'{module} does not emit a structured line'
    assert "unreadable, continuing without it: {e}" not in src, (
        f'{module} still prints the raw exception, which embeds the absolute path')
    assert f"'{module}'" in src, f'{module} does not identify itself in its log line'


def test_real_module_read_failure_line_is_clean(tmp_path, monkeypatch, capsys):
    """End-to-end: corrupt the file a module actually reads, check what it prints."""
    import statcast_features
    cache_file = tmp_path / 'deep' / 'cache_statcast.csv'
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text('key,v\n"unterminated\n\x00binary', encoding='utf-8')
    monkeypatch.setattr(statcast_features, 'CACHE_FILE', str(cache_file))
    monkeypatch.setattr(statcast_features, '_MEM_CACHE', {})

    statcast_features._load_cache()
    out = capsys.readouterr().out
    assert '[cache-read-fail]' in out
    assert 'module=statcast_features' in out
    assert 'file=cache_statcast.csv' in out
    assert 'deep' not in out, 'directory name leaked'
    assert str(tmp_path) not in out, 'absolute path leaked'


# ── the real modules use the safe path ──────────────────────────────────────

@pytest.mark.parametrize('module', ['statcast_features', 'bvp_stats', 'bullpen_data'])
def test_modules_no_longer_write_csv_directly(module):
    src = open(os.path.join(REPO, f'{module}.py'), encoding='utf-8').read()
    assert '.to_csv(' not in src, (
        f'{module} writes CSV directly again — that bypasses the lock and the '
        'atomic replace')
    assert 'atomic_cache' in src


@pytest.mark.parametrize('module', ['statcast_features', 'bvp_stats', 'bullpen_data'])
def test_modules_have_no_unguarded_mutate_then_save(module):
    """`cache[k] = v` followed by `_save_cache(cache)` is the original bug."""
    import re
    src = open(os.path.join(REPO, f'{module}.py'), encoding='utf-8').read()
    assert not re.search(r'cache\[\w+\]\s*=\s*\w+\s*\n\s*_save_cache\(', src), (
        f'{module} reintroduced mutate-then-save outside the lock')
