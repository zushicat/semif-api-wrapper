"""Golden integration test: Jev request -> SemIf row -> MLX -> Jev response
(implementation-rev-1 §9, plan-rev-1 §6 Step 9).

Run explicitly:  pytest -q -m integration

The fixtures in docs/Semif/ capture a prior live run of the same pinned model
(decisions.jsonl = rows as served, results-mlx-direct.jsonl = scorer output).
Each fixture row becomes one Jev request with a single choice question; the
response must reproduce the captured probabilities to 4 decimals, the captured
argmax, and the captured real token count — pinning the whole pipeline
translator -> prompt -> MLX -> response mapping end-to-end.

Skips cleanly when SEMIF_MODEL is unset or the checkpoint path is missing.
"""

import json
from pathlib import Path

import pytest

from semif_runtime import runtime
from settings import config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not config.SEMIF_MODEL or not Path(config.SEMIF_MODEL).exists(),
        reason=f"SEMIF_MODEL not set or checkpoint missing: {config.SEMIF_MODEL!r}",
    ),
]

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "docs" / "Semif"


def _read_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row
            for line in path.read_text().splitlines() if line.strip()
            for row in (json.loads(line),)}


DECISIONS = _read_jsonl(_FIXTURE_DIR / "decisions.jsonl")
RESULTS = _read_jsonl(_FIXTURE_DIR / "results-mlx-direct.jsonl")


@pytest.fixture(scope="module")
def live_model():
    """Load the real model once for the module (ASGITransport does not run
    FastAPI lifespans, so the lifespan preload never fires in tests).

    Pins SEMIF_MODE=direct by direct assignment (a function-scoped monkeypatch
    inside a module-scoped fixture is a ScopeMismatch): this golden is a
    *direct-mode* baseline, so the suite must be .env-independent."""
    config.SEMIF_MODE = "direct"      # pin: this golden is a direct baseline
    runtime.load()
    yield runtime
    runtime.reset()
    config.SEMIF_MODE = "direct"      # restore (pydantic models are mutable)


def _fixture_request(decision: dict) -> dict:
    """One fixture row -> one Jev request (single choice question)."""
    return {
        "state": decision["state"],                       # verbatim str
        "model": "jev-latest",
        "questions": {decision["id"]: {
            "type": "choice",
            "instructions": decision["question"],
            "criteria": {o["id"]: o["description"] for o in decision["options"]},
        }},
    }


def _fixture_argmax(decision: dict, result: dict) -> str:
    probabilities = result["probabilities"]
    return result["option_ids"][max(range(len(probabilities)),
                                    key=probabilities.__getitem__)]


@pytest.mark.parametrize("decision_id", sorted(DECISIONS))
async def test_golden_row_reproduces_fixture(client, live_model, decision_id):
    decision = DECISIONS[decision_id]
    result = RESULTS[decision_id]

    response = await client.post("/v1/systemone", json=_fixture_request(decision))

    assert response.status_code == 200, response.text
    body = response.json()
    answer = body["answers"][decision_id]

    # 4-decimal agreement per option (builder rounding is a no-op here: the
    # captured probabilities round to a map that already sums to exactly 1).
    expected = {o["id"]: round(p, 4)
                for o, p in zip(decision["options"], result["probabilities"])}
    assert answer["probabilities"] == expected

    # argmax option id == the fixture's argmax
    assert answer["choice"] == _fixture_argmax(decision, result)

    # real token counts: one question -> one row
    assert body["usage"]["input_tokens"] == result["input_tokens"]
    assert body["usage"]["output_tokens"] == 0

    # model echo untouched
    assert body["model"] == "jev-latest"