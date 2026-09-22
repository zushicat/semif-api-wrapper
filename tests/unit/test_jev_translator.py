"""Unit tests for the SystemOneRequest -> SemIf rows translator (impl §5).

One row per question in insertion order; state passes verbatim; rows are
checked with SemIf's own validate_row so bad shapes raise SemIfValidationError
(-> 422) before any model work.
"""

import json

import pytest

from classes.jev_errors import SemIfValidationError
from classes.jev_translator import build_semif_rows, serialize_text
from models.jev_models import ChoiceQuestion, NoulQuestion, SystemOneRequest


def make_request(questions: dict, state="test state") -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {"state": state, "model": "jev-latest", "questions": questions})


def options_of(request, question_id="q"):
    rows = build_semif_rows(request)
    row = next(r for r in rows if r["id"] == question_id)
    return {o["id"]: o["description"] for o in row["options"]}


# 1. noul, no criteria -> Yes/No fallbacks.
def test_noul_without_criteria_gets_yes_no_defaults():
    request = make_request({"q": {"type": "noul", "instructions": "Is this a request?"}})
    assert options_of(request) == {"true": "Yes", "false": "No"}


def test_noul_with_explicit_null_criteria_gets_yes_no_defaults():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?",
                                  "criteria": None}})
    assert options_of(request) == {"true": "Yes", "false": "No"}


# 2. noul, both criteria parts -> descriptions verbatim.
def test_noul_both_criteria_parts_verbatim():
    request = make_request({
        "q": {"type": "noul", "instructions": "Is this?",
              "criteria": {"true": "A request", "false": "Not a request"}}})
    assert options_of(request) == {"true": "A request", "false": "Not a request"}


# 3. noul, one part present -> the other falls back.
def test_noul_partial_criteria_fall_back():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?",
                                  "criteria": {"true": "A request"}}})
    assert options_of(request) == {"true": "A request", "false": "No"}

    request = make_request({"q": {"type": "noul", "instructions": "Is this?",
                                  "criteria": {"false": "Not a request"}}})
    assert options_of(request) == {"true": "Yes", "false": "Not a request"}


# 4. noul criteria part "" -> empty after stringify -> fallback.
def test_noul_empty_string_criteria_falls_back():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?",
                                  "criteria": {"true": "", "false": "  "}}})
    assert options_of(request) == {"true": "Yes", "false": "No"}


# 5. noul criteria part as dict -> JSON-stringified description.
def test_noul_dict_criteria_json_stringified():
    request = make_request({"q": {"type": "noul", "instructions": "Is this?",
                                  "criteria": {"true": {"yes_means": "help"},
                                               "false": {"no_means": "chat"}}}})
    options = options_of(request)
    assert options["true"] == json.dumps({"yes_means": "help"}, ensure_ascii=False, indent=2)
    assert options["false"] == json.dumps({"no_means": "chat"}, ensure_ascii=False, indent=2)


# 6. choice: null rubrics -> placeholder; string rubrics verbatim.
def test_choice_null_rubrics_get_placeholder():
    request = make_request({
        "q": {"type": "choice", "instructions": "pick",
              "criteria": {"CAT": None, "DOG": "A dog"}}})
    assert options_of(request) == {"CAT": "(no additional description)",
                                   "DOG": "A dog"}


def test_choice_empty_string_rubric_stays_verbatim():
    request = make_request({
        "q": {"type": "choice", "instructions": "pick",
              "criteria": {"A": "", "B": "x"}}})
    assert options_of(request)["A"] == ""


def test_choice_option_ids_may_contain_whitespace_and_any_length():
    request = make_request({
        "q": {"type": "choice", "instructions": "pick",
              "criteria": {"a long option id with spaces": None, "x" * 40: None}}})
    row = build_semif_rows(request)[0]
    assert [o["id"] for o in row["options"]] == [
        "a long option id with spaces", "x" * 40]


# 7. choice keys appear as option ids in dict insertion order.
def test_choice_keys_preserve_insertion_order():
    request = make_request({
        "q": {"type": "choice", "instructions": "pick",
              "criteria": {"zebra": None, "alpha": None, "mid": None}}})
    row = build_semif_rows(request)[0]
    assert [o["id"] for o in row["options"]] == ["zebra", "alpha", "mid"]


# 8. score: option ids "0".."N-1", descriptions = serialize_text(level).
def test_score_levels_become_indexed_options():
    request = make_request({
        "q": {"type": "score", "instructions": "rate",
              "criteria": ["Calm", "Frustrated", {"very": "angry"}]}})
    row = build_semif_rows(request)[0]
    assert [o["id"] for o in row["options"]] == ["0", "1", "2"]
    assert row["options"][0]["description"] == "Calm"
    assert row["options"][1]["description"] == "Frustrated"
    assert row["options"][2]["description"] == json.dumps(
        {"very": "angry"}, ensure_ascii=False, indent=2)


# 9. state passes through verbatim: dict/list keep value and type (pydantic
#    copies dicts/lists during validation, so identity holds only when the
#    request is built with model_construct); never stringified.
def test_state_passes_through_by_identity():
    state = {"customer": "frustrated", "history": [1, 2, 3]}
    request = make_request(
        {"q": {"type": "noul", "instructions": "Is this?"}}, state=state)
    row = build_semif_rows(request)[0]
    assert row["state"] == state
    assert isinstance(row["state"], dict)          # not stringified
    assert row["state"]["history"] == [1, 2, 3]

    state_list = ["line one", "line two"]
    request = make_request(
        {"q": {"type": "noul", "instructions": "Is this?"}}, state=state_list)
    row = build_semif_rows(request)[0]
    assert row["state"] == state_list
    assert isinstance(row["state"], list)

    request = make_request(
        {"q": {"type": "noul", "instructions": "Is this?"}}, state="plain text")
    assert build_semif_rows(request)[0]["state"] == "plain text"


def test_state_is_the_same_object_when_bypassing_validation():
    # model_construct assigns fields verbatim -> the row references the exact
    # object the translator received (no copy, no stringify step anywhere).
    state = {"customer": "frustrated"}
    request = SystemOneRequest.model_construct(
        state=state, model="jev-latest",
        questions={"q": NoulQuestion.model_validate(
            {"type": "noul", "instructions": "Is this?"})})
    assert build_semif_rows(request)[0]["state"] is state


# 10. row order equals questions insertion order; ids = question ids.
def test_row_order_and_ids_follow_questions():
    request = make_request({
        "s1": {"type": "score", "instructions": "rate", "criteria": ["a", "b"]},
        "n1": {"type": "noul", "instructions": "Is this?"},
        "c1": {"type": "choice", "instructions": "pick", "criteria": {"A": None, "B": None}},
    })
    rows = build_semif_rows(request)
    assert [r["id"] for r in rows] == ["s1", "n1", "c1"]


# 11. question text = serialized instructions (str verbatim; dict -> JSON).
def test_question_text_is_serialized_instructions():
    request = make_request({"q": {"type": "noul", "instructions": "Is this a request?"}})
    assert build_semif_rows(request)[0]["question"] == "Is this a request?"

    request = make_request({
        "q": {"type": "noul", "instructions": {"what": "request?", "lang": "en"}}})
    assert build_semif_rows(request)[0]["question"] == json.dumps(
        {"what": "request?", "lang": "en"}, ensure_ascii=False, indent=2)


# 12. SemIf-rejected rows raise SemIfValidationError naming the question id.
#     NaN inside a dict state bypasses pydantic (model_construct) but is
#     caught by validate_row's finite-JSON check -> 422, not a scorer 500.
def test_nan_state_rejected_as_validation_error():
    request = SystemOneRequest.model_construct(
        state={"x": float("nan")},
        model="jev-latest",
        questions={"q": NoulQuestion.model_validate(
            {"type": "noul", "instructions": "Is this?"})})
    with pytest.raises(SemIfValidationError, match="'q'") as excinfo:
        build_semif_rows(request)
    assert "finite JSON-compatible" in str(excinfo.value)


def test_validate_row_rejection_wraps_question_id():
    # Noul always emits the valid true/false pair, so violate the 2-option
    # floor instead: a model_construct'd 1-option choice skips pydantic's
    # bounds check and fails validate_row (2 <= len(options)).
    request = SystemOneRequest.model_construct(
        state="state",
        model="jev-latest",
        questions={"q": ChoiceQuestion.model_construct(
            type="choice", instructions="pick", criteria={"only": None})})
    with pytest.raises(SemIfValidationError, match="'q'"):
        build_semif_rows(request)