"""Shared pytest fixtures (implementation-rev-1 §7.3).

- ``_forbid_real_model_load``: autouse guard — unit tests must never touch
  the real model; integration tests opt out via the ``integration`` marker.
- ``client``: ASGI in-process client for the FastAPI app (no live server;
  ASGITransport does not run lifespans, which is fine because the fake
  runtime replaces scoring anyway).
- ``make_result``: canned SemIf result row aligned with a translated row.
- ``fake_semif``: installs a fake runtime; tests populate
  ``.results_by_id`` before the call.
- ``request_factory``: minimal valid SystemOneRequest payloads.

The real model can only be reached by tests marked ``integration`` (they call
``runtime.load()`` themselves — the ASGI test client never runs lifespans).
"""

import threading

import pytest
from httpx import ASGITransport, AsyncClient

from app import app
from models.jev_models import SystemOneRequest
from semif_phase1 import mlx_backend
from semif_runtime import runtime


@pytest.fixture(autouse=True)
def _forbid_real_model_load(request, monkeypatch):
    """Unit tests must never touch the real model; integration tests opt out
    entirely (they load/reset the runtime in their own module fixture)."""
    if request.node.get_closest_marker("integration"):
        yield
        return

    runtime.reset()

    def _boom(*_a, **_k):
        raise AssertionError("real model load attempted in a unit test")

    # Guard the heavy operations, not runtime.load: the runtime's own
    # config-validation paths are legitimately unit-tested.
    monkeypatch.setattr(mlx_backend, "load_model", _boom)
    monkeypatch.setattr(mlx_backend, "score", _boom)
    yield
    runtime.reset()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=None) as c:
        yield c


def make_result(row: dict, probabilities: list[float], input_tokens: int = 100) -> dict:
    """Canned SemIf result row aligned with `row` (used by response tests too)."""
    return {"id": row["id"],
            "option_ids": [o["id"] for o in row["options"]],
            "probabilities": probabilities,
            "input_tokens": input_tokens,
            "option_logits": [0.0] * len(probabilities)}


class _FakeRuntime:
    """Records scored rows and the cancel flag; returns canned results."""
    def __init__(self, results_by_id: dict[str, dict]):
        self.results_by_id = results_by_id
        self.scored: list[list[dict]] = []
        self.cancel_flags: list[threading.Event | None] = []

    def score_rows(self, rows, cancelled=None):
        self.scored.append(rows)
        self.cancel_flags.append(cancelled)
        return [self.results_by_id[row["id"]] for row in rows]


@pytest.fixture
def fake_semif(monkeypatch):
    """Install a fake runtime; tests populate `.results_by_id` before the call."""
    fake = _FakeRuntime(results_by_id={})
    monkeypatch.setattr(runtime, "score_rows", fake.score_rows, raising=True)
    return fake


@pytest.fixture
def request_factory():
    def _make(**overrides) -> SystemOneRequest:
        payload = {"state": "test state", "model": "jev-latest",
                   "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}}}
        payload.update(overrides)
        return SystemOneRequest(**payload)
    return _make