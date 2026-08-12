"""
Thread-safe, crash-safe CSV-backed dict caches.

Three modules (statcast_features, bvp_stats, bullpen_data) each kept a
module-level `_MEM_CACHE` and returned it **by reference** from their loader.
Callers then did `cache[key] = result` and `_save_cache(cache)`, and the saver
iterated that same shared dict:

    for k, v in cache.items()          # thread B, serialising
    cache[key] = result                # thread A, mutating

Under the worker's 8-thread pool that raises
`RuntimeError: dictionary changed size during iteration`, which the worker
recorded as an unexpected failure and dropped the batter. Seven batters were
lost in the 2026-08-12 18:32 production run; the same error appears in the
17:02, 17:31 and 18:01 runs. Before the funnel existed it was swallowed by
`except RuntimeError: return None  # insufficient history` and mislabelled as a
sparse-history batter.

Snapshotting the dict alone is NOT sufficient. It removes the iteration error
but leaves two failure modes:

  * concurrent `to_csv` calls on one path interleave or truncate — last writer
    wins and intermediate updates vanish;
  * a reader landing mid-write gets a partial file, and the old
    `except Exception: return {}` turned that into a silent, total cache loss
    that looked like ordinary cold-cache behaviour.

So: one lock per cache file guarding mutation, snapshot, serialisation and
replacement; a temp file in the destination directory; flush + fsync before
close; `os.replace` for an atomic swap; temp cleanup on failure; and the
existing cache left intact when a write fails.

SCOPE BOUNDARY — read this before relying on it.
`threading.Lock` serialises threads **inside one process**. It does not
coordinate separate Railway services, separate processes, or multiple replicas
writing one filesystem. That is acceptable here only because these are local
derived caches on ephemeral per-service disks: `web` and the cron worker each
have their own copy, nothing is shared between them, and a lost cache entry
costs one refetch. `os.replace` still guarantees a reader never sees a torn
file even across processes. Any state that must be correct *across* services
belongs in Postgres, not here.
"""
import os
import threading

import pandas as pd


class CacheCorruptionError(RuntimeError):
    """The cache file exists but could not be parsed.

    Distinct from 'no cache yet'. The old code returned {} for both, so a
    corrupt or half-written file silently discarded every entry and looked
    identical to a cold start.
    """


# One lock per absolute cache path. Two modules pointed at the same file would
# share a lock, which is the required behaviour.
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def lock_for(path: str) -> threading.Lock:
    key = os.path.abspath(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.RLock()
        return lock


def load_cache(path: str, key_col: str) -> dict:
    """Read a CSV cache into {key: {col: val}}.

    Returns {} when the file does not exist — a genuine cold start.
    Raises CacheCorruptionError when it exists but cannot be parsed.
    """
    if not os.path.exists(path):
        return {}
    with lock_for(path):
        try:
            df = pd.read_csv(path, dtype={key_col: str})
        except Exception as e:
            raise CacheCorruptionError(
                f'{path} exists but could not be parsed: {type(e).__name__}: {e}'
            ) from e
        if df.empty or key_col not in df.columns:
            return {}
        return df.set_index(key_col).to_dict('index')


def save_cache(path: str, key_col: str, cache: dict) -> None:
    """Serialise `cache` to `path` atomically, under the path's lock.

    The snapshot is taken while holding the lock, so no other thread can mutate
    the dict mid-iteration. The write goes to a temp file in the SAME directory
    (os.replace is only atomic within a filesystem), is flushed and fsynced
    before close, then replaced into place. On any failure the temp file is
    removed and the previous cache file is left untouched.
    """
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(directory, exist_ok=True)

    with lock_for(path):
        # Snapshot under the lock — this is what stops the iteration error.
        rows = [{key_col: k, **v} for k, v in cache.items()]

        tmp = os.path.join(directory, f'.{os.path.basename(path)}.{os.getpid()}.'
                                      f'{threading.get_ident()}.tmp')
        try:
            with open(tmp, 'w', newline='', encoding='utf-8') as fh:
                pd.DataFrame(rows).to_csv(fh, index=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)          # atomic; readers see old or new
        except Exception:
            # Leave the existing cache intact and drop the partial temp file.
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            raise


def describe_failure(kind: str, module: str, path: str, exc: BaseException,
                     action: str) -> str:
    """One structured, path-safe line for a cache failure.

        [cache-read-fail] module=statcast_features file=cache_statcast.csv
                          error=ParserError action=refetch

    Only the basename is emitted. The raw exception text is deliberately NOT
    included: CacheCorruptionError embeds the absolute path, and these lines
    land in Railway logs visible to anyone with project access. The module,
    file and exception class are what make a failure actionable; the parent
    directory adds nothing diagnostic.

    Reports the underlying cause where there is one, so a wrapped parse error
    shows as ParserError rather than the CacheCorruptionError wrapper.
    """
    cause = exc.__cause__ if exc.__cause__ is not None else exc
    return (f'  [{kind}] module={module} file={os.path.basename(path)} '
            f'error={type(cause).__name__} action={action}')


def save_cache_best_effort(path: str, key_col: str, cache: dict,
                           module: str = 'unknown') -> bool:
    """Persist, but never let a write failure destroy the caller's result.

    A cache write is an optimisation: the value is already in memory and is
    being returned regardless, so the only cost of a failed write is a refetch
    later. Letting it raise would propagate out of get_batter_statcast, through
    run_prediction, into the worker's `except Exception` — recorded as an
    unexpected failure and the batter dropped. That is the outcome this whole
    change exists to prevent.

    Distinct from a failed READ, which must NOT be swallowed: returning {} for
    a corrupt file silently discards every entry, so load_cache still raises.
    Loud here, fatal there.
    """
    try:
        save_cache(path, key_col, cache)
        return True
    except Exception as e:
        print(describe_failure('cache-write-fail', module, path, e,
                               'continue-in-memory'))
        return False


def update_cache(path: str, key_col: str, mem: dict, key, value) -> None:
    """Mutate the shared in-memory dict and persist it, both under one lock.

    Callers must not do `cache[key] = v; save_cache(...)` themselves — that
    reopens the exact window this module exists to close.
    """
    with lock_for(path):
        mem[key] = value
        save_cache(path, key_col, mem)


def cleanup_stale_temps(path: str) -> int:
    """Remove temp files abandoned by a crashed writer. Returns how many."""
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    prefix = f'.{os.path.basename(path)}.'
    removed = 0
    try:
        for name in os.listdir(directory):
            if name.startswith(prefix) and name.endswith('.tmp'):
                try:
                    os.remove(os.path.join(directory, name))
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed
