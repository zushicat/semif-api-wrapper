"""Unit tests for the SemIf result -> SystemOneResponse builder (impl §6).

Fixtures: Jev requests + rows built by the real translator (cross-checks the
Step 4 contract) + canned SemIf result rows. Once Step 6 lands, ``make_result``
moves to tests/conftest.py; it lives locally here until that rework.
"""

import json

import pytest

from classes.jev_errors import SemIfRuntimeError
from classes.jev_response_builder import (
    _argmax,
    _confidence,
    _distribution,
    _renormalize,
    build_systemone_response,
    scoring_summary,
)
from classes.jev_translator import build_semif_rows
from models.jev_models import SystemOneRequest
from settings import config


def make_request(questions: dict) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {"state": "test state", "model": "jev-latest", "questions": questions})


def make_result(row: dict, probabilities: list[float], input_tokens: int = 100) -> dict:
    """Canned SemIf result row aligned with `row` (conftest helper from Step 6)."""
    return {"id": row["id"],
            "option_ids": [o["id"] for o in row["options"]],
            "probabilities": probabilities,
            "input_tokens": input_tokens,
            "option_logits": [0.0] * len(probabilities)}


def build(request, probabilities_by_id, input_tokens_by_id=None):
    """Translate -> fake-score -> build, the same pipeline the route runs."""
    rows = build_semif_rows(request)
    input_tokens_by_id = input_tokens_by_id or {}
    results = [make_result(row, probabilities_by_id[row["id"]],
                           input_tokens_by_id.get(row["id"], 100))
               for row in rows]
    return build_systemone_response(request, rows, results)


# --------------------------------------------------------------------------
# 1/2. noul
# --------------------------------------------------------------------------


def test_noul_probability_from_true_rounded():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?"}})
    response = build(request, {"q": [0.9991704935, 0.0008295065]})
    assert response.answers["q"].noul == 0.9992


def test_noul_carries_no_confidence_field():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?"}})
    response = build(request, {"q": [0.87, 0.13]})
    assert "confidence" not in response.answers["q"].model_dump()
    assert response.answers["q"].type == "noul"


# --------------------------------------------------------------------------
# 3-7. choice
# --------------------------------------------------------------------------


def make_choice_request(criteria):
    return make_request({"q": {"type": "choice", "instructions": "pick",
                               "criteria": criteria}})


def test_choice_probabilities_declared_order_rounded_sum_exactly_one():
    request = make_choice_request({"CAT": None, "BIRD": None, "DOG": None})
    response = build(request, {"q": [0.999312345, 0.000612345, 0.00017531]})
    probs = response.answers["q"].probabilities
    assert list(probs) == ["CAT", "BIRD", "DOG"]          # declared order
    assert all(round(v, 4) == v for v in probs.values())  # rounded
    assert sum(probs.values()) == 1.0                     # exactly


def test_choice_rounding_residual_goes_to_argmax():
    request = make_choice_request({"A": None, "B": None, "C": None})
    response = build(request, {"q": [1 / 3, 1 / 3, 1 / 3]})
    probs = response.answers["q"].probabilities
    assert probs == {"A": 0.3334, "B": 0.3333, "C": 0.3333}  # residual 0.0001 -> argmax A
    assert sum(probs.values()) == 1.0


def test_choice_confidence_published_formula():
    request = make_choice_request({"A": None, "B": None})
    # two options, p_max = 0.87 -> (2*0.87 - 1) / 1 = 0.74
    response = build(request, {"q": [0.87, 0.13]})
    assert response.answers["q"].confidence == 0.74


def test_choice_confidence_all_mass_and_uniform():
    request = make_choice_request({"A": None, "B": None, "C": None})
    response = build(request, {"q": [1.0, 0.0, 0.0]})
    assert response.answers["q"].confidence == 1.0
    # exactly uniform in 4 decimals: no residual, peak p_max = 1/n
    request = make_choice_request({"A": None, "B": None})
    response = build(request, {"q": [0.5, 0.5]})
    assert response.answers["q"].confidence == 0.0


def test_choice_argmax_and_tie_goes_to_first_declared_key():
    request = make_choice_request({"A": None, "B": None, "C": None})
    response = build(request, {"q": [0.2, 0.5, 0.3]})
    assert response.answers["q"].choice == "B"

    tie = build(request, {"q": [0.5, 0.5, 0.0]})
    assert tie.answers["q"].choice == "A"  # first declared key on ties


def test_choice_choice_value_is_argmax_of_final_distribution():
    # Residual shifts A to 0.3334, but the argmax is still C at 0.6000.
    request = make_choice_request({"A": None, "B": None, "C": None})
    response = build(request, {"q": [0.3333, 0.0667, 0.6000]})
    assert response.answers["q"].choice == "C"


# --------------------------------------------------------------------------
# 8/9. score
# --------------------------------------------------------------------------


def make_score_request(criteria):
    return make_request({"q": {"type": "score", "instructions": "rate",
                               "criteria": criteria}})


def test_score_expectation_over_final_distribution():
    request = make_score_request(["Calm", "Frustrated", "Very angry"])
    response = build(request, {"q": [0.1, 0.6, 0.3]})
    answer = response.answers["q"]
    assert answer.score == round(0 * 0.1 + 1 * 0.6 + 2 * 0.3, 4)  # 1.2: between levels
    assert list(answer.probabilities) == ["0", "1", "2"]
    assert sum(answer.probabilities.values()) == 1.0


def test_score_legend_verbatim_and_serialized():
    request = make_score_request(["Calm", {"frustrated": True}, ["very", "angry"]])
    response = build(request, {"q": [0.5, 0.3, 0.2]})
    legend = response.answers["q"].legend
    assert list(legend) == ["0", "1", "2"]
    assert legend["0"] == "Calm"
    assert legend["1"] == json.dumps({"frustrated": True}, ensure_ascii=False, indent=2)
    assert legend["2"] == json.dumps(["very", "angry"], ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# 10/11. model echo and usage
# --------------------------------------------------------------------------


def test_model_echoes_request_by_default():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?"}})
    response = build(request, {"q": [0.87, 0.13]})
    assert response.model == "jev-latest"


def test_reported_model_id_overrides(monkeypatch):
    monkeypatch.setattr(config, "REPORTED_MODEL_ID", "jev-proxy/semif-mlx")
    request = make_request({"q": {"type": "noul", "instructions": "Is this?"}})
    response = build(request, {"q": [0.87, 0.13]})
    assert response.model == "jev-proxy/semif-mlx"


def test_usage_sums_real_row_token_counts():
    request = make_request({
        "n": {"type": "noul", "instructions": "Is this?"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"A": None, "B": None}},
    })
    response = build(request,
                     {"n": [0.87, 0.13], "c": [0.6, 0.4]},
                     input_tokens_by_id={"n": 142, "c": 130})
    assert response.usage.input_tokens == 272
    assert response.usage.output_tokens == 0


# --------------------------------------------------------------------------
# 12. misalignment -> SemIfRuntimeError (the proxy never invents answers)
# --------------------------------------------------------------------------


def test_results_count_mismatch_raises():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?"}})
    rows = build_semif_rows(request)
    with pytest.raises(SemIfRuntimeError, match="wrong number of result rows"):
        build_systemone_response(request, rows, [])


def test_result_id_mismatch_raises():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?"}})
    rows = build_semif_rows(request)
    result = make_result(rows[0], [0.87, 0.13])
    result["id"] = "other"
    with pytest.raises(SemIfRuntimeError, match="'q'"):
        build_systemone_response(request, rows, [result])


def test_option_ids_mismatch_raises():
    request = make_choice_request({"A": None, "B": None})
    rows = build_semif_rows(request)
    result = make_result(rows[0], [0.5, 0.5])
    result["option_ids"] = ["X", "Y"]          # swapped ids, right lengths
    with pytest.raises(SemIfRuntimeError, match="does not match declared options"):
        build_systemone_response(request, rows, [result])


def test_option_ids_probabilities_length_mismatch_raises():
    request = make_choice_request({"A": None, "B": None})
    rows = build_semif_rows(request)
    result = make_result(rows[0], [0.5, 0.5])
    result["probabilities"] = [0.5]
    with pytest.raises(SemIfRuntimeError, match="misaligned"):
        build_systemone_response(request, rows, [result])


def test_missing_result_for_question_raises():
    request = make_request({
        "q": {"type": "noul", "instructions": "Is this?"},
        "r": {"type": "noul", "instructions": "And this?"},
    })
    rows = build_semif_rows(request)
    # count matches but both results claim id "q" -> r has no result
    results = [make_result(rows[0], [0.87, 0.13]),
               make_result(rows[0], [0.90, 0.10])]
    with pytest.raises(SemIfRuntimeError, match="missing for question 'r'"):
        build_systemone_response(request, rows, results)


# --------------------------------------------------------------------------
# 5 (adapted). strict alignment supersedes the defensive 0.0
# --------------------------------------------------------------------------


def test_declared_option_missing_from_result_raises():
    # _distribution's strict `got != option_ids` check makes the 0.0 default
    # unreachable by design: a result lacking a declared option is
    # misalignment (-> 500), never a silently-invented 0.0.
    request = make_choice_request({"A": None, "B": None})
    rows = build_semif_rows(request)
    result = make_result(rows[0], [1.0])
    result["option_ids"] = ["A"]               # B missing
    with pytest.raises(SemIfRuntimeError, match="does not match declared options"):
        _distribution(result, ["A", "B"], "q")


def test_distribution_maps_declared_options():
    result = {"option_ids": ["A", "B"], "probabilities": [0.25, 0.75]}
    assert _distribution(result, ["A", "B"], "q") == {"A": 0.25, "B": 0.75}


# --------------------------------------------------------------------------
# 13 (dropped). single-option confidence test — bounds make n=1 impossible
# --------------------------------------------------------------------------


def test_argmax_prefers_first_max_key():
    assert _argmax({"a": 0.5, "b": 0.5, "c": 0.0}) == "a"
    assert _argmax({"a": 0.1, "b": 0.9}) == "b"


# --------------------------------------------------------------------------
# 14. scoring_summary (Fix 4, implementation-rev-2 §4.1): per-request timing
#     summary for the response headers; rows without timing -> None
# --------------------------------------------------------------------------


def _timing_result(row_id: str, total: float, forward: float | None) -> dict:
    """Canned SemIf row with the timing fields real scorer rows carry."""
    result = make_result({"id": row_id,
                          "options": [{"id": "true"}, {"id": "false"}]},
                         [0.87, 0.13])
    result["total_seconds"] = total
    if forward is not None:
        result["forward_seconds"] = forward
    result["model"] = {"serving_config": "mlx-direct-v1"}
    return result


def test_scoring_summary_sums_timing_counts_rows_and_mode():
    results = [_timing_result("a", 0.5, 0.4), _timing_result("b", 0.25, 0.2)]
    assert scoring_summary(results) == {"scoring_ms": 750, "forward_ms": 600,
                                        "rows": 2, "mode": "mlx-direct-v1"}


def test_scoring_summary_none_without_timing_fields():
    # current make_result shape (no total_seconds) -> None -> no headers
    request = make_request({"q": {"type": "noul", "instructions": "Is this?"}})
    rows = build_semif_rows(request)
    assert scoring_summary([make_result(rows[0], [0.87, 0.13])]) is None


def test_scoring_summary_forward_missing_gives_none_forward_ms():
    summary = scoring_summary([_timing_result("a", 0.5, None)])
    assert summary == {"scoring_ms": 500, "forward_ms": None,
                       "rows": 1, "mode": "mlx-direct-v1"}