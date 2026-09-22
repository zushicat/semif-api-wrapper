"""Unit tests for the in-process SemIf scoring runtime (implementation §3).

No test here may touch the real model: config is monkeypatched or the runtime
is primed with fakes, and an autouse fixture resets the singleton around each
test so state never leaks between cases.
"""

import inspect
import math
import threading

import pytest

import semif_runtime
from classes.jev_errors import (SemIfCancelled, SemIfRuntimeError,
                                SemIfValidationError)
from semif_phase1 import mlx_backend
from semif_runtime import _checked, _wrap_scoring_error, runtime
from settings import Settings, config


@pytest.fixture(autouse=True)
def _fresh_runtime():
    """Isolate the module-level singleton around every test."""
    runtime.reset()
    yield
    runtime.reset()


@pytest.fixture
def fake_loader(monkeypatch):
    """Replace mlx_backend.load_model with a counting fake."""
    calls = []

    def _fake_load_model(source, revision, bits=None, *, cache_limit_mib=None):
        calls.append({"source": source, "revision": revision, "bits": bits,
                      "cache_limit_mib": cache_limit_mib})
        return ("model", "tokenizer", "metadata")

    monkeypatch.setattr(mlx_backend, "load_model", _fake_load_model)
    return calls


def _prime_runtime(monkeypatch, score_fn):
    """Prime the singleton as loaded with a fake `_score_one`
    (the monkeypatch-priming pattern shared by the score_rows tests)."""
    monkeypatch.setattr(runtime, "_loaded", True)
    monkeypatch.setattr(runtime, "_model", "model")
    monkeypatch.setattr(runtime, "_tokenizer", "tokenizer")
    monkeypatch.setattr(runtime, "_metadata", "metadata")
    monkeypatch.setattr(runtime, "_score_one", score_fn)


# 1. load() with SEMIF_MODEL="" -> SemIfRuntimeError naming SEMIF_MODEL,
#    raised before any mlx import (the check precedes the import in load()).
def test_load_aborts_without_model(monkeypatch):
    monkeypatch.setattr(config, "SEMIF_MODEL", "")
    with pytest.raises(SemIfRuntimeError, match="SEMIF_MODEL"):
        runtime.load()
    assert runtime._loaded is False


# 2. load() with SEMIF_BACKEND="torch" -> SemIfRuntimeError naming SEMIF_BACKEND.
def test_load_rejects_torch_backend(monkeypatch):
    monkeypatch.setattr(config, "SEMIF_BACKEND", "torch")
    with pytest.raises(SemIfRuntimeError, match="SEMIF_BACKEND"):
        runtime.load()
    assert runtime._loaded is False


# 3. load() with SEMIF_MODE="shared" -> "not implemented yet".
def test_load_rejects_shared_mode(monkeypatch):
    monkeypatch.setattr(config, "SEMIF_MODE", "shared")
    with pytest.raises(SemIfRuntimeError, match="not implemented yet"):
        runtime.load()
    assert runtime._loaded is False


def test_load_rejects_unknown_mode(monkeypatch):
    monkeypatch.setattr(config, "SEMIF_MODE", "turbo")
    with pytest.raises(SemIfRuntimeError, match="unknown SEMIF_MODE"):
        runtime.load()
    assert runtime._loaded is False


# 4. load() is idempotent: two load() calls -> one loader call, with the
#    configured model/revision forwarded (bits None, cache default 256).
def test_load_is_idempotent(monkeypatch, fake_loader):
    monkeypatch.setattr(config, "SEMIF_MODE", "direct")   # pin: .env-independent
    runtime.load()
    runtime.load()
    assert len(fake_loader) == 1
    assert fake_loader[0]["source"] == config.SEMIF_MODEL
    assert fake_loader[0]["revision"] == config.SEMIF_REVISION
    assert fake_loader[0]["bits"] is None
    assert fake_loader[0]["cache_limit_mib"] == mlx_backend.DEFAULT_CACHE_LIMIT_MIB
    assert runtime._loaded is True
    assert runtime._score_one is mlx_backend.score


def test_load_passes_optional_cache_limit(monkeypatch, fake_loader):
    monkeypatch.setattr(config, "SEMIF_MODE", "direct")   # pin: .env-independent
    monkeypatch.setattr(config, "SEMIF_MLX_CACHE_LIMIT_MIB", 512)
    runtime.load()
    assert fake_loader[0]["cache_limit_mib"] == 512


# 5. reset() then load() -> the loader runs again.
def test_reset_then_load_reloads(monkeypatch, fake_loader):
    monkeypatch.setattr(config, "SEMIF_MODE", "direct")   # pin: .env-independent
    runtime.load()
    runtime.reset()
    assert runtime._loaded is False
    runtime.load()
    assert len(fake_loader) == 2


# 6. score_rows calls the scorer once per row, in order, with the configured
#    max token count; results come back aligned with the rows.
def test_score_rows_calls_score_per_row(monkeypatch):
    calls = []

    def _fake_score(model, tokenizer, row, metadata, max_tokens):
        calls.append((row["id"], max_tokens))
        return {"probabilities": [1.0, 0.0]}

    monkeypatch.setattr(runtime, "_loaded", True)
    monkeypatch.setattr(runtime, "_model", "model")
    monkeypatch.setattr(runtime, "_tokenizer", "tokenizer")
    monkeypatch.setattr(runtime, "_metadata", "metadata")
    monkeypatch.setattr(runtime, "_score_one", _fake_score)

    rows = [{"id": "a", "state": {}}, {"id": "b", "state": {}}]
    results = runtime.score_rows(rows)

    assert calls == [("a", config.SEMIF_MAX_TOKENS), ("b", config.SEMIF_MAX_TOKENS)]
    assert [r["probabilities"] for r in results] == [[1.0, 0.0], [1.0, 0.0]]


# 6b. A scorer ValueError is classified by _wrap_scoring_error with the
#     failing row's id (over-limit -> 422-shaped validation error).
def test_score_rows_wraps_over_limit_valueerror(monkeypatch):
    def _fake_score(model, tokenizer, row, metadata, max_tokens):
        if row["id"] == "second":
            raise ValueError(f"Row {row['id']}: {max_tokens + 1} input tokens "
                             f"exceed limit {max_tokens}; no truncation allowed")
        return {"probabilities": [1.0, 0.0]}

    monkeypatch.setattr(runtime, "_loaded", True)
    monkeypatch.setattr(runtime, "_model", "model")
    monkeypatch.setattr(runtime, "_tokenizer", "tokenizer")
    monkeypatch.setattr(runtime, "_metadata", "metadata")
    monkeypatch.setattr(runtime, "_score_one", _fake_score)

    with pytest.raises(SemIfValidationError, match="second"):
        runtime.score_rows([{"id": "first", "state": {}}, {"id": "second", "state": {}}])


# 6c. Cancellation (Fix 1, implementation-rev-2 §2.2): the cancel event is
#     checked under the lock *before* each row, so an abandoned request wastes
#     at most the one row already in flight.


# 1. Event preset before scoring -> SemIfCancelled, _score_one never called.
def test_score_rows_cancelled_before_start(monkeypatch):
    calls = []

    def _fake_score(model, tokenizer, row, metadata, max_tokens):
        calls.append(row["id"])
        return {"probabilities": [1.0, 0.0]}

    _prime_runtime(monkeypatch, _fake_score)
    event = threading.Event()
    event.set()

    rows = [{"id": "a", "state": {}}, {"id": "b", "state": {}}]
    with pytest.raises(SemIfCancelled) as excinfo:
        runtime.score_rows(rows, cancelled=event)

    assert calls == []                      # no GPU work for the abandoned request
    assert excinfo.value.rows_scored == 0
    assert excinfo.value.rows_total == 2
    assert "0/2" in str(excinfo.value)


# 2. Event set after the first row -> second row never scored, progress 1/2.
def test_score_rows_cancelled_midway(monkeypatch):
    calls = []
    event = threading.Event()

    def _fake_score(model, tokenizer, row, metadata, max_tokens):
        calls.append(row["id"])
        if len(calls) == 1:
            event.set()                     # client disconnects after row 1
        return {"probabilities": [1.0, 0.0]}

    _prime_runtime(monkeypatch, _fake_score)

    with pytest.raises(SemIfCancelled) as excinfo:
        runtime.score_rows([{"id": "a", "state": {}}, {"id": "b", "state": {}}],
                           cancelled=event)

    assert calls == ["a"]
    assert excinfo.value.rows_scored == 1
    assert excinfo.value.rows_total == 2


# 3. No flag (default) -> unchanged behavior: all rows scored, results aligned.
#    Pins the signature default `cancelled=None`.
def test_score_rows_without_flag_scores_all(monkeypatch):
    default = inspect.signature(runtime.score_rows).parameters["cancelled"].default
    assert default is None

    calls = []

    def _fake_score(model, tokenizer, row, metadata, max_tokens):
        calls.append(row["id"])
        return {"probabilities": [1.0, 0.0]}

    _prime_runtime(monkeypatch, _fake_score)

    results = runtime.score_rows([{"id": "a", "state": {}}, {"id": "b", "state": {}}])

    assert calls == ["a", "b"]
    assert [r["probabilities"] for r in results] == [[1.0, 0.0], [1.0, 0.0]]


# 4. The lock is acquired once per row (per-row cycles, was one hold for the
#    whole request). A real lock is necessarily held *during* each row, so the
#    per-row behavior is observable as re-acquisition.
def test_score_rows_acquires_lock_per_row(monkeypatch):
    class _CountingLock:
        def __init__(self):
            self.acquisitions = 0

        def __enter__(self):
            self.acquisitions += 1
            return self

        def __exit__(self, *exc_info):
            return False

    counting = _CountingLock()
    monkeypatch.setattr(semif_runtime, "_lock", counting)
    _prime_runtime(monkeypatch, lambda *_a: {"probabilities": [1.0, 0.0]})

    runtime.score_rows([{"id": "a", "state": {}}, {"id": "b", "state": {}}])

    assert counting.acquisitions == 2


# 6d. Serial mode (Fix 2, implementation-rev-2 §3.2): SEMIF_MODE=serial loads,
#     rows route through a stateful scorer created once per score_rows call —
#     the prefix cache must never be shared across requests.


# 1. SEMIF_MODE="serial" is accepted; the mode is bound at load and the
#    _score_one binding is unchanged.
def test_load_accepts_serial_mode(monkeypatch, fake_loader):
    monkeypatch.setattr(config, "SEMIF_MODE", "serial")
    runtime.load()
    assert runtime._loaded is True
    assert runtime._mode == "serial"
    assert runtime._score_one is mlx_backend.score


# 2. test_load_rejects_shared_mode (above) is unchanged: the shared rejection
#    still precedes the accepted-set check, so its message stays pinned.


# 3. Serial rows route through the per-request scorer in declared order with
#    the configured max-token count; the constructor got the real
#    model/tokenizer/metadata tuple.
def test_serial_routes_rows_through_scorer(monkeypatch, fake_loader):
    made = []
    instances = []

    class _FakeScorer:
        def __init__(self, model, tokenizer, metadata, max_tokens):
            made.append((model, tokenizer, metadata, max_tokens))
            instances.append(self)
            self.scored = []

        def score(self, row):
            self.scored.append(row["id"])
            return {"probabilities": [1.0, 0.0]}

    monkeypatch.setattr(config, "SEMIF_MODE", "serial")
    monkeypatch.setattr(mlx_backend, "SerialPrefixScorer", _FakeScorer)
    runtime.load()

    results = runtime.score_rows([{"id": "a", "state": {}}, {"id": "b", "state": {}}])

    assert made == [("model", "tokenizer", "metadata", config.SEMIF_MAX_TOKENS)]
    assert instances[0].scored == ["a", "b"]
    assert [r["probabilities"] for r in results] == [[1.0, 0.0], [1.0, 0.0]]


# 4. Two score_rows calls -> two scorer instances (per-request lifetime; the
#    prefix cache is not shared across requests).
def test_serial_scorer_is_per_request(monkeypatch, fake_loader):
    instances = []

    class _FakeScorer:
        def __init__(self, model, tokenizer, metadata, max_tokens):
            instances.append(self)
            self.scored = []

        def score(self, row):
            self.scored.append(row["id"])
            return {"probabilities": [1.0, 0.0]}

    monkeypatch.setattr(config, "SEMIF_MODE", "serial")
    monkeypatch.setattr(mlx_backend, "SerialPrefixScorer", _FakeScorer)
    runtime.load()

    rows = [{"id": "a", "state": {}}, {"id": "b", "state": {}}]
    runtime.score_rows(rows)
    runtime.score_rows(rows)

    assert len(instances) == 2
    assert instances[0].scored == ["a", "b"]
    assert instances[1].scored == ["a", "b"]


# 5. The cancel flag also stops the serial scorer between rows.
def test_serial_cancelled_flag_stops_scorer(monkeypatch, fake_loader):
    event = threading.Event()
    scored = []

    class _FakeScorer:
        def __init__(self, model, tokenizer, metadata, max_tokens):
            pass

        def score(self, row):
            scored.append(row["id"])
            if len(scored) == 1:
                event.set()                 # client disconnects after row 1
            return {"probabilities": [1.0, 0.0]}

    monkeypatch.setattr(config, "SEMIF_MODE", "serial")
    monkeypatch.setattr(mlx_backend, "SerialPrefixScorer", _FakeScorer)
    runtime.load()

    with pytest.raises(SemIfCancelled) as excinfo:
        runtime.score_rows([{"id": "a", "state": {}}, {"id": "b", "state": {}}],
                           cancelled=event)

    assert scored == ["a"]
    assert excinfo.value.rows_scored == 1
    assert excinfo.value.rows_total == 2


# 6. Direct mode creates no scorer: _new_scorer() is None and the plain
#    _score_one path is used (existing test_score_rows_calls_score_per_row
#    doubles as the direct-path regression pin).
def test_direct_mode_has_no_scorer(monkeypatch, fake_loader):
    monkeypatch.setattr(config, "SEMIF_MODE", "direct")   # pin: .env-independent
    runtime.load()
    assert runtime._mode == "direct"
    assert runtime._new_scorer() is None


# 7. _wrap_scoring_error: over-limit ValueError -> validation error;
#    any other ValueError (e.g. slot-token) -> runtime error.
def test_wrap_scoring_error_over_limit_is_validation():
    exc = ValueError("Row r1: 5000 input tokens exceed limit 4096; no truncation allowed")
    wrapped = _wrap_scoring_error(exc, "r1")
    assert isinstance(wrapped, SemIfValidationError)
    assert "r1" in str(wrapped)
    assert "SEMIF_MAX_TOKENS" in str(wrapped)


def test_wrap_scoring_error_other_valueerror_is_runtime():
    exc = ValueError("Answer slot 'A' is not one exact round-trip token")
    wrapped = _wrap_scoring_error(exc, "r1")
    assert isinstance(wrapped, SemIfRuntimeError)
    assert "r1" in str(wrapped)


# 8. _checked: malformed output is rejected; the proxy never invents answers.
def test_checked_rejects_missing_probabilities():
    with pytest.raises(SemIfRuntimeError, match="malformed"):
        _checked({"id": "r1", "option_ids": ["true", "false"]})


def test_checked_rejects_nan_and_inf():
    for bad in ([float("nan"), 0.5], [0.5, float("inf")], [0.5, float("-inf")]):
        with pytest.raises(SemIfRuntimeError, match="non-finite"):
            _checked({"probabilities": bad})
    assert math.isnan(float("nan"))  # NaN comparison sanity for the guard above


def test_checked_passes_valid_result():
    result = {"probabilities": [0.75, 0.25], "option_ids": ["true", "false"]}
    assert _checked(result) is result


# 9. Settings plumbing: "" placeholders for the optional ints coerce to None
#    (pydantic-settings would otherwise reject them).
def test_empty_to_none_coerces():
    from settings import _empty_to_none
    assert _empty_to_none("") is None
    assert _empty_to_none("   ") is None
    assert _empty_to_none("4") == "4"


def test_settings_empty_int_placeholders_become_none():
    s = Settings(SEMIF_MLX_BITS="", SEMIF_MLX_CACHE_LIMIT_MIB="")
    assert s.SEMIF_MLX_BITS is None
    assert s.SEMIF_MLX_CACHE_LIMIT_MIB is None


def test_settings_int_values_parse():
    s = Settings(SEMIF_MLX_BITS="4", SEMIF_MLX_CACHE_LIMIT_MIB="512")
    assert s.SEMIF_MLX_BITS == 4
    assert s.SEMIF_MLX_CACHE_LIMIT_MIB == 512