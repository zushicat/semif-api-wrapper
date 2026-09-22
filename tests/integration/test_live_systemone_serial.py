"""Serial-mode golden: same fixtures as the direct golden, pinned against
docs/Semif/results-mlx-serial.jsonl (generated via semif-score --mode serial;
see implementation-rev-2 §3.4). Forces SEMIF_MODE=serial by direct assignment
so the suite is .env-independent — serial probabilities differ from direct in
late decimals (documented drift), so it must not run under whichever mode
.env names.

Run explicitly:  pytest -q -m integration

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
RESULTS = _read_jsonl(_FIXTURE_DIR / "results-mlx-serial.jsonl")


@pytest.fixture(scope="module")
def live_model():
    """Load the real model once for the module, pinned to serial mode."""
    config.SEMIF_MODE = "serial"      # pin: this golden is a serial baseline
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

    # 4-decimal agreement per option against the *serial* baseline (identical
    # assertion discipline to the direct golden).
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
