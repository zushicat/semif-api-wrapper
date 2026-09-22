"""Route tests through the app with a fake SemIf runtime (impl §7.3).

Covers: happy-path shape, the three answer types, 422/500 error mapping,
model allowlist, auth (401/403), GET /v1/models, and legacy 404s. No HTTP
mocking anywhere: scoring is replaced at the runtime boundary.
"""

import threading

import pytest

import routes.jev_route
from classes.jev_errors import (SemIfCancelled, SemIfRuntimeError,
                                SemIfValidationError)
from classes.jev_translator import build_semif_rows
from models.jev_models import SystemOneRequest
from settings import config


def make_payload(**overrides) -> dict:
    payload = {"state": "test state", "model": "jev-latest",
               "questions": {"q": {"type": "noul", "instructions": "Is this a test?"}}}
    payload.update(overrides)
    return payload


def make_result(row: dict, probabilities: list[float], input_tokens: int = 100) -> dict:
    """Canned SemIf result row aligned with `row` (mirror of the conftest helper)."""
    return {"id": row["id"],
            "option_ids": [o["id"] for o in row["options"]],
            "probabilities": probabilities,
            "input_tokens": input_tokens,
            "option_logits": [0.0] * len(probabilities)}


def seed_results(request_payload: dict, probabilities_by_id, input_tokens_by_id=None):
    """Translate the payload and build aligned canned results for fake_semif."""
    input_tokens_by_id = input_tokens_by_id or {}
    request = SystemOneRequest.model_validate(request_payload)
    return {row["id"]: make_result(row, probabilities_by_id[row["id"]],
                                   input_tokens_by_id.get(row["id"], 100))
            for row in build_semif_rows(request)}


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


async def test_happy_path_noul_returns_jev_shape(client, fake_semif):
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))

    response = await client.post("/v1/systemone", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "jev-latest"
    assert body["answers"]["q"] == {"type": "noul", "noul": 0.87}
    assert body["usage"] == {"input_tokens": 100, "output_tokens": 0}

    # the runtime received exactly one row, with the state verbatim
    assert len(fake_semif.scored) == 1
    (rows,) = fake_semif.scored
    assert len(rows) == 1
    assert rows[0]["id"] == "q"
    assert rows[0]["state"] == "test state"


async def test_mixed_question_types_return_all_answer_shapes(client, fake_semif):
    payload = make_payload(questions={
        "n": {"type": "noul", "instructions": "Is this?"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"CAT": None, "DOG": None}},
        "s": {"type": "score", "instructions": "rate", "criteria": ["Calm", "Angry"]},
    })
    fake_semif.results_by_id.update(seed_results(
        payload,
        {"n": [0.9, 0.1], "c": [0.6, 0.4], "s": [0.25, 0.75]},
        input_tokens_by_id={"n": 100, "c": 80, "s": 62}))

    response = await client.post("/v1/systemone", json=payload)

    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["n"]["type"] == "noul"
    assert answers["n"]["noul"] == 0.9
    assert answers["c"]["type"] == "choice"
    assert answers["c"]["choice"] == "CAT"
    assert answers["c"]["probabilities"] == {"CAT": 0.6, "DOG": 0.4}
    assert answers["c"]["confidence"] == 0.2  # (2*0.6 - 1) / 1
    assert answers["s"]["type"] == "score"
    assert answers["s"]["score"] == 0.75      # 0*0.25 + 1*0.75
    assert answers["s"]["legend"] == {"0": "Calm", "1": "Angry"}
    assert response.json()["usage"] == {"input_tokens": 242, "output_tokens": 0}


async def test_extra_request_fields_are_ignored(client, fake_semif):
    payload = make_payload(metadata={"request_id": "sdk-abc"})
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200


async def test_reported_model_id_env_is_respected(client, fake_semif, monkeypatch):
    monkeypatch.setattr(config, "REPORTED_MODEL_ID", "jev-proxy-semif-1.0.0")
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))
    response = await client.post("/v1/systemone", json=payload)
    assert response.json()["model"] == "jev-proxy-semif-1.0.0"


async def test_any_model_accepted_without_allowlist(client, fake_semif):
    payload = make_payload(model="anything-goes")
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    assert response.json()["model"] == "anything-goes"


# --------------------------------------------------------------------------
# 422 — request validation (default FastAPI body) and SemIf validation errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {**make_payload(), "state": 7},  # scalar state
        {**make_payload(), "state": ""},  # empty state (Step 3 validator)
        {k: v for k, v in make_payload().items() if k != "state"},  # missing state
        {**make_payload(), "model": ""},  # empty model
        {**make_payload(), "questions": {}},  # empty questions map
        {
            **make_payload(),
            "questions": {"q": {"type": "score", "instructions": "rate", "criteria": ["only"]}},
        },  # score with 1 level
        {
            **make_payload(),
            "questions": {"q": {"type": "unknown", "instructions": "?"}},
        },  # unknown type
        {
            **make_payload(),
            "questions": {"q": {"type": "choice", "instructions": "pick",
                                "criteria": {f"opt{i}": None for i in range(17)}}},
        },  # 17-option choice above the SemIf bound
    ],
)
async def test_validation_failures_return_default_422_body(client, payload):
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)  # FastAPI's default validation body


async def test_semif_validation_error_maps_to_422_with_detail(client, monkeypatch):
    from semif_runtime import runtime

    def _boom(rows, cancelled=None):
        raise SemIfValidationError("Question 'q': state must be finite JSON-compatible data")

    monkeypatch.setattr(runtime, "score_rows", _boom, raising=True)
    response = await client.post("/v1/systemone", json=make_payload())
    assert response.status_code == 422
    assert "finite JSON-compatible" in response.json()["detail"]


async def test_not_allowed_model_returns_422(client, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_MODELS", "jev-latest, jev-preview")
    response = await client.post("/v1/systemone", json=make_payload(model="gpt-4o"))
    assert response.status_code == 422
    assert "not allowed" in response.json()["detail"]


async def test_allowed_model_accepted_when_allowlist_set(client, fake_semif, monkeypatch):
    monkeypatch.setattr(config, "ALLOWED_MODELS", "jev-latest, jev-preview")
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 200


# --------------------------------------------------------------------------
# 500 — runtime failures (the proxy never invents answers)
# --------------------------------------------------------------------------


async def test_semif_runtime_error_maps_to_500_with_detail(client, monkeypatch):
    from semif_runtime import runtime

    def _boom(rows, cancelled=None):
        raise SemIfRuntimeError("Scorer returned a malformed result row")

    monkeypatch.setattr(runtime, "score_rows", _boom, raising=True)
    response = await client.post("/v1/systemone", json=make_payload())
    assert response.status_code == 500
    assert "malformed result row" in response.json()["detail"]


# --------------------------------------------------------------------------
# 499 — client disconnected before/during scoring (Fix 1, impl-rev-2 §2.3)
# --------------------------------------------------------------------------


async def test_cancelled_request_returns_499(client, monkeypatch):
    from semif_runtime import runtime

    def _abandoned(rows, cancelled=None):
        raise SemIfCancelled(1, 2)

    monkeypatch.setattr(runtime, "score_rows", _abandoned, raising=True)
    response = await client.post("/v1/systemone", json=make_payload())
    assert response.status_code == 499      # bare response; no detail body contract


async def test_route_passes_cancel_flag_to_runtime(client, fake_semif):
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))

    response = await client.post("/v1/systemone", json=payload)

    assert response.status_code == 200
    (flag,) = fake_semif.cancel_flags
    assert isinstance(flag, threading.Event)
    assert not flag.is_set()                # a fully served request was never abandoned


async def test_watch_disconnect_sets_event(monkeypatch):
    from routes.jev_route import _watch_disconnect

    class _StubRequest:
        """`is_disconnected()` replays a scripted answer sequence."""
        def __init__(self, answers):
            self.answers = list(answers)

        async def is_disconnected(self):
            return self.answers.pop(0)

    # Disconnect on the first poll -> event set, coroutine returns.
    event = threading.Event()
    await _watch_disconnect(_StubRequest([True]), event)
    assert event.is_set()

    # Polling loop: two misses, then the disconnect (interval pinned small —
    # we pin the loop, not the timing).
    monkeypatch.setattr(routes.jev_route, "_CANCEL_POLL_S", 0.001)
    event = threading.Event()
    await _watch_disconnect(_StubRequest([False, False, True]), event)
    assert event.is_set()


# --------------------------------------------------------------------------
# Timing headers (Fix 4, implementation-rev-2 §4.2): informational only — the
# Jev body contract is untouched
# --------------------------------------------------------------------------


_TIMING_HEADERS = ("X-Request-Ms", "X-Scoring-Ms", "X-Scoring-Rows",
                   "X-Scoring-Forward-Ms", "X-Scoring-Mode")


async def test_timing_headers_present_with_timing_fakes(client, fake_semif):
    payload = make_payload()
    results = seed_results(payload, {"q": [0.87, 0.13]})
    for result in results.values():
        result["total_seconds"] = 0.5
        result["forward_seconds"] = 0.4
        result["model"] = {"serving_config": "mlx-direct-v1"}
    fake_semif.results_by_id.update(results)

    response = await client.post("/v1/systemone", json=payload)

    assert response.status_code == 200
    headers = response.headers
    assert headers["X-Scoring-Ms"] == "500"           # Σ row totals
    assert headers["X-Scoring-Rows"] == "1"
    assert headers["X-Scoring-Forward-Ms"] == "400"
    assert headers["X-Scoring-Mode"] == "mlx-direct-v1"
    for name in ("X-Request-Ms", "X-Scoring-Ms", "X-Scoring-Rows",
                 "X-Scoring-Forward-Ms"):             # numeric -> integer strings
        assert headers[name].isdigit(), name
    # the Jev body contract is untouched
    assert set(response.json()) == {"model", "answers", "usage"}


async def test_timing_headers_absent_with_plain_fakes(client, fake_semif):
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))

    response = await client.post("/v1/systemone", json=payload)

    assert response.status_code == 200
    for name in _TIMING_HEADERS:
        assert name not in response.headers
    # response body unchanged (Jev contract untouched)
    body = response.json()
    assert set(body) == {"model", "answers", "usage"}
    assert body["answers"]["q"] == {"type": "noul", "noul": 0.87}
    assert body["usage"] == {"input_tokens": 100, "output_tokens": 0}


async def test_validation_error_has_no_timing_headers(client, monkeypatch):
    from semif_runtime import runtime

    def _boom(rows, cancelled=None):
        raise SemIfValidationError("Question 'q': state must be finite JSON-compatible data")

    monkeypatch.setattr(runtime, "score_rows", _boom, raising=True)
    response = await client.post("/v1/systemone", json=make_payload())
    assert response.status_code == 422
    for name in _TIMING_HEADERS:
        assert name not in response.headers


# --------------------------------------------------------------------------
# Auth: 401 missing / 401 bad scheme / 403 wrong token
# --------------------------------------------------------------------------


@pytest.fixture
def auth_enabled(monkeypatch):
    monkeypatch.setattr(config, "USE_API_KEY", True)
    monkeypatch.setattr(config, "API_KEY", "sk-1234")


async def test_missing_authorization_header_returns_401(client, fake_semif, auth_enabled):
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))
    response = await client.post("/v1/systemone", json=payload)
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing Authorization header"}


async def test_invalid_scheme_returns_401(client, fake_semif, auth_enabled):
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))
    response = await client.post(
        "/v1/systemone", json=payload, headers={"Authorization": "Basic sk-1234"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authorization scheme"}


async def test_wrong_token_returns_403(client, fake_semif, auth_enabled):
    response = await client.post(
        "/v1/systemone", json=make_payload(), headers={"Authorization": "Bearer sk-wrong"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid token"}


async def test_correct_token_returns_200(client, fake_semif, auth_enabled):
    payload = make_payload()
    fake_semif.results_by_id.update(seed_results(payload, {"q": [0.87, 0.13]}))
    response = await client.post(
        "/v1/systemone", json=payload, headers={"Authorization": "Bearer sk-1234"})
    assert response.status_code == 200


async def test_health_endpoint_stays_unauthenticated(client, auth_enabled):
    response = await client.get("/")
    assert response.status_code == 200


# --------------------------------------------------------------------------
# GET /v1/models and unknown paths
# --------------------------------------------------------------------------


async def test_get_models_returns_static_catalog(client):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"models"}
    names = [model["name"] for model in body["models"]]
    assert names == ["jev-latest", "jev-preview"]
    for model in body["models"]:
        assert set(model) == {"name", "description", "release_date"}
    assert "SemIf" in body["models"][0]["description"]


async def test_unknown_path_returns_default_404(client):
    response = await client.post("/nonexistent", json={})
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}