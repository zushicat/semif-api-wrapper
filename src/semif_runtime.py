"""In-process SemIf scoring runtime: one model, loaded once, lock-serialized.

MLX models and the Metal GPU are not thread-safe; the lock gives requests the
same serialize-behind-a-GPU-lock behavior the remote engine had. The lock
serializes *rows*, not *requests*: each row is scored under its own lock
cycle, so abandoned requests (client disconnected; the route's watcher sets a
cancel event) skip their remaining rows instead of holding the GPU for their
full scoring time. The route awaits asyncio.to_thread(...) so the event loop
stays free while scoring.
"""

import threading

from classes.jev_errors import SemIfCancelled, SemIfRuntimeError, SemIfValidationError
from settings import config

_lock = threading.Lock()


def _wrap_scoring_error(exc: Exception, row_id: str) -> Exception:
    """Client-caused ValueErrors -> 422; everything else -> 500."""
    text = str(exc)
    if "input tokens exceed limit" in text:      # encode_prompt's exact message
        return SemIfValidationError(
            f"Prompt for decision {row_id!r} is too long ({text}); "
            "shorten the state/instructions or raise SEMIF_MAX_TOKENS")
    return SemIfRuntimeError(f"Scoring failed for decision {row_id!r}: {text}")


class SemIfRuntime:
    def __init__(self) -> None:
        self._loaded = False
        self._model = self._tokenizer = self._metadata = None
        self._score_one = None   # mlx_backend.score once loaded
        self._mode = "direct"    # bound at load(); direct until then

    def load(self) -> None:
        """Idempotent; blocks until the model is ready. Raises
        SemIfRuntimeError with the offending settings name on bad config."""
        if self._loaded:
            return
        with _lock:
            if self._loaded:
                return
            if not config.SEMIF_MODEL or not config.SEMIF_REVISION:
                raise SemIfRuntimeError(
                    "SEMIF_MODEL and SEMIF_REVISION must be set in .env "
                    f"(got SEMIF_MODEL={config.SEMIF_MODEL!r}, "
                    f"SEMIF_REVISION={config.SEMIF_REVISION!r})")
            if config.SEMIF_BACKEND != "mlx":
                raise SemIfRuntimeError(
                    f"SEMIF_BACKEND {config.SEMIF_BACKEND!r} is not supported "
                    "(this revision ships mlx only)")
            if config.SEMIF_MODE == "shared":
                raise SemIfRuntimeError("SEMIF_MODE=shared is not implemented yet")  # later rev
            if config.SEMIF_MODE not in ("direct", "serial"):
                raise SemIfRuntimeError(f"unknown SEMIF_MODE {config.SEMIF_MODE!r}")
            from semif_phase1 import mlx_backend   # heavy import happens here
            self._model, self._tokenizer, self._metadata = mlx_backend.load_model(
                config.SEMIF_MODEL, config.SEMIF_REVISION, config.SEMIF_MLX_BITS,
                cache_limit_mib=(config.SEMIF_MLX_CACHE_LIMIT_MIB
                                 if config.SEMIF_MLX_CACHE_LIMIT_MIB is not None
                                 else mlx_backend.DEFAULT_CACHE_LIMIT_MIB))
            self._score_one = mlx_backend.score
            self._mode = config.SEMIF_MODE          # bound at load, read by score_rows
            self._loaded = True

    def score_rows(self, rows: list[dict],
                   cancelled: threading.Event | None = None) -> list[dict]:
        """Score N rows in declared order; results are aligned with rows.

        `cancelled` is a threading.Event set by the route's disconnect watcher
        (analysis-rev-2 §5 Fix 1). It is checked under the lock *before* each row,
        so an abandoned request wastes at most the one row already in flight —
        a Python thread cannot be killed mid-`mx.eval` (A §1.2, §2).

        Serial mode: all rows of a Jev request share the exact same state, so
        the stateful SerialPrefixScorer is created once per score_rows call.
        The cache is keyed on exact prefix token ids, so per-request instances
        are safe and nothing leaks across requests.
        """
        self.load()
        scorer = self._new_scorer()
        results: list[dict] = []
        for row in rows:
            with _lock:                    # per-row lock cycle (was: one hold for all rows)
                if cancelled is not None and cancelled.is_set():
                    raise SemIfCancelled(len(results), len(rows))
                try:
                    results.append(self._score_row(row, scorer))
                except ValueError as exc:
                    raise _wrap_scoring_error(exc, row["id"]) from exc
        return [_checked(result) for result in results]

    def _new_scorer(self):
        """Serial mode: one stateful prefix-cache scorer per request."""
        if self._mode != "serial":
            return None
        from semif_phase1 import mlx_backend      # already imported by load()
        return mlx_backend.SerialPrefixScorer(self._model, self._tokenizer,
                                              self._metadata, config.SEMIF_MAX_TOKENS)

    def _score_row(self, row, scorer):
        if scorer is not None:
            return scorer.score(row)
        return self._score_one(self._model, self._tokenizer, row,
                               self._metadata, config.SEMIF_MAX_TOKENS)

    def reset(self) -> None:
        """Drop the model (shutdown / tests)."""
        with _lock:
            self._model = self._tokenizer = self._metadata = None
            self._score_one = None
            self._mode = "direct"    # a primed-but-unloaded runtime behaves like direct
            self._loaded = False


def _checked(result: dict) -> dict:
    """The proxy never invents answers: reject malformed scorer output."""
    if not isinstance(result, dict) or "probabilities" not in result:
        raise SemIfRuntimeError("Scorer returned a malformed result row")
    if any(not isinstance(p, (int, float)) or p != p or p in (float("inf"), float("-inf"))
           for p in result["probabilities"]):
        raise SemIfRuntimeError("Scorer returned non-finite probabilities")
    return result


runtime = SemIfRuntime()