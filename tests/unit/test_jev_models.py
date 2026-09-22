"""Unit tests for the Jev contract models (Phase 2 acceptance, plan §4/§6).

Every validation failure must surface as a pydantic ``ValidationError``, which
FastAPI turns into the default 422 body (Jev semantics).
"""

import pytest
from pydantic import ValidationError

from models.jev_models import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

VALID_REQUEST = {
    "state": "My cat has food allergies. Does your pet food contain any allergens?",
    "model": "jev-latest",
    "questions": {
        "is_request": {"type": "noul", "instructions": "Is this a request?"},
        "department": {
            "type": "choice",
            "instructions": "What kind of pet does the customer have?",
            "criteria": {"CAT": None, "BIRD": None, "DOG": None},
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def parse_valid_request():
    return SystemOneRequest.model_validate(VALID_REQUEST)


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_valid_three_type_request_parses():
    request = parse_valid_request()

    assert request.model == "jev-latest"
    assert request.state == VALID_REQUEST["state"]

    assert isinstance(request.questions["is_request"], NoulQuestion)
    assert isinstance(request.questions["department"], ChoiceQuestion)
    assert isinstance(request.questions["frustration"], ScoreQuestion)

    # Choice criteria: insertion order preserved, None rubrics kept as None.
    assert list(request.questions["department"].criteria) == ["CAT", "BIRD", "DOG"]
    assert request.questions["department"].criteria == {"CAT": None, "BIRD": None, "DOG": None}

    # Score criteria: levels verbatim.
    assert request.questions["frustration"].criteria == ["Calm", "Frustrated", "Very angry"]

    # Noul criteria: optional, absent here.
    assert request.questions["is_request"].criteria is None


def test_question_type_discriminators_bind_correctly():
    request = parse_valid_request()
    assert request.questions["is_request"].type == "noul"
    assert request.questions["department"].type == "choice"
    assert request.questions["frustration"].type == "score"


def test_state_accepts_string_dict_and_list():
    for state in ["plain text", {"customer": "frustrated"}, ["line one", "line two"]]:
        payload = {**VALID_REQUEST, "state": state}
        assert SystemOneRequest.model_validate(payload).state == state


def test_instructions_accept_dict_and_list():
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {"type": "noul", "instructions": {"what": "request?", "lang": "en"}}
        },
    }
    question = SystemOneRequest.model_validate(payload).questions["q"]
    assert question.instructions == {"what": "request?", "lang": "en"}


def test_noul_criteria_parts_are_optional():
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {
                "type": "noul",
                "instructions": "Is this a request?",
                "criteria": {"true": "A request"},
            }
        },
    }
    question = SystemOneRequest.model_validate(payload).questions["q"]
    assert question.criteria is not None
    assert question.criteria.true == "A request"
    assert question.criteria.false is None


def test_extra_fields_are_ignored():
    payload = {
        **VALID_REQUEST,
        "metadata": {"client": "typesafe-sdk", "request_id": "abc"},  # SDK additions
        "questions": {
            "q": {"type": "noul", "instructions": "ok", "user_data": {"x": 1}}
        },
    }
    request = SystemOneRequest.model_validate(payload)
    assert list(request.questions) == ["q"]


def test_questions_insertion_order_preserved():
    request = parse_valid_request()
    assert list(request.questions) == ["is_request", "department", "frustration"]


# --------------------------------------------------------------------------
# Validation failures -> 422 (plan §2.8 matrix)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", [7, 3.14, True, None])
def test_scalar_state_rejected(state):
    payload = {**VALID_REQUEST, "state": state}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("missing", ["state", "model", "questions"])
def test_missing_required_fields_rejected(missing):
    payload = {k: v for k, v in VALID_REQUEST.items() if k != missing}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_empty_model_rejected():
    payload = {**VALID_REQUEST, "model": ""}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_empty_questions_rejected():
    payload = {**VALID_REQUEST, "questions": {}}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_unknown_question_type_rejected():
    payload = {
        **VALID_REQUEST,
        "questions": {"q": {"type": "boolean", "instructions": "old scaffold type"}},
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("options", [0, 1, 17, 255])
def test_choice_criteria_out_of_bounds_rejected(options):
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {"type": "choice", "instructions": "pick", "criteria": {} if options == 0
                  else {f"opt{i}": None for i in range(options)}}
        },
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("options", [2, 16])
def test_choice_criteria_bounds_accepted(options):
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {f"opt{i}": None for i in range(options)},
            }
        },
    }
    question = SystemOneRequest.model_validate(payload).questions["q"]
    assert len(question.criteria) == options


def test_choice_criteria_error_message_names_both_bounds():
    """The message becomes the 422 detail via pydantic; both bounds must show."""
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {"type": "choice", "instructions": "pick",
                  "criteria": {"only": None}}
        },
    }
    with pytest.raises(ValidationError, match="between 2 and 16 options"):
        SystemOneRequest.model_validate(payload)


# --------------------------------------------------------------------------
# state must not be empty (SemIf rejects it; surface as clean 422)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["", "   ", "\t\n", {}, []])
def test_empty_state_rejected(state):
    payload = {**VALID_REQUEST, "state": state}
    with pytest.raises(ValidationError, match="state must not be empty"):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("state", [{"a": 1}, [1, 2], [" "], {"": ""}])
def test_state_with_content_accepted(state):
    payload = {**VALID_REQUEST, "state": state}
    assert SystemOneRequest.model_validate(payload).state == state


@pytest.mark.parametrize("levels", [1, 11])
def test_score_criteria_out_of_bounds_rejected(levels):
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {
                "type": "score",
                "instructions": "rate",
                "criteria": [f"level{i}" for i in range(levels)],
            }
        },
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("levels", [2, 10])
def test_score_criteria_bounds_accepted(levels):
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {
                "type": "score",
                "instructions": "rate",
                "criteria": [f"level{i}" for i in range(levels)],
            }
        },
    }
    question = SystemOneRequest.model_validate(payload).questions["q"]
    assert len(question.criteria) == levels


@pytest.mark.parametrize("instructions", ["", "   "])
def test_empty_instructions_rejected(instructions):
    payload = {
        **VALID_REQUEST,
        "questions": {"q": {"type": "noul", "instructions": instructions}},
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_score_level_must_be_stringifiable():
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {"type": "score", "instructions": "rate", "criteria": ["ok", None]}
        },
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_choice_rubric_must_be_string_or_null():
    payload = {
        **VALID_REQUEST,
        "questions": {
            "q": {"type": "choice", "instructions": "pick", "criteria": {"A": 7}}
        },
    }
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------


def test_answer_models_parse():
    noul = NoulAnswer.model_validate({"type": "noul", "noul": 0.0291})
    choice = ChoiceAnswer.model_validate(
        {
            "type": "choice",
            "choice": "CAT",
            "probabilities": {"CAT": 0.9993, "BIRD": 0.0006, "DOG": 0.0001},
            "confidence": 0.999,
        }
    )
    score = ScoreAnswer.model_validate(
        {
            "type": "score",
            "score": 0.4019,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.7506, "1": 0.0969, "2": 0.1525},
            "confidence": 0.6259,
        }
    )

    assert noul.noul == 0.0291
    assert choice.choice == "CAT"
    assert score.score == 0.4019


def test_answer_unknown_type_rejected():
    with pytest.raises(ValidationError):
        NoulAnswer.model_validate({"type": "boolean", "value": True})


def test_answer_probability_bounds_enforced():
    with pytest.raises(ValidationError):
        NoulAnswer.model_validate({"type": "noul", "noul": 1.5})
    with pytest.raises(ValidationError):
        ChoiceAnswer.model_validate(
            {"type": "choice", "choice": "A", "probabilities": {"A": 0.5}, "confidence": 2.0}
        )


def test_full_response_round_trips():
    response = SystemOneResponse.model_validate(
        {
            "model": "jev-latest",
            "answers": {
                "is_request": {"type": "noul", "noul": 0.0291},
                "department": {
                    "type": "choice",
                    "choice": "CAT",
                    "probabilities": {"CAT": 0.9993, "BIRD": 0.0006, "DOG": 0.0001},
                    "confidence": 0.999,
                },
                "frustration": {
                    "type": "score",
                    "score": 0.4019,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.7506, "1": 0.0969, "2": 0.1525},
                    "confidence": 0.6259,
                },
            },
            "usage": {"input_tokens": 74, "output_tokens": 0},
        }
    )
    assert isinstance(response.answers["is_request"], NoulAnswer)
    assert isinstance(response.answers["department"], ChoiceAnswer)
    assert isinstance(response.answers["frustration"], ScoreAnswer)
    assert response.usage == Usage(input_tokens=74, output_tokens=0)

    # Serialize back: the response shape is JSON-ready for the route.
    dumped = response.model_dump()
    assert SystemOneResponse.model_validate(dumped) == response


def test_usage_tokens_must_be_non_negative():
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1, output_tokens=0)
