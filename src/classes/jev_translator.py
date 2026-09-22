"""SystemOneRequest -> SemIf decision rows (plan-rev-1 §4 D1).

One row per question, in questions insertion order. state passes through
verbatim (str/dict/list). Rows are validated with SemIf's own validate_row so
row-shape problems surface as 422 before any model work.
"""

import json

from classes.jev_errors import SemIfValidationError
from models.jev_models import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)
from semif_phase1.core import validate_row

_NO_ADDITIONAL_DESCRIPTION = "(no additional description)"


def serialize_text(value) -> str:
    """``str`` verbatim; everything else JSON-encoded."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _rubric_or(value, fallback: str) -> str:
    """criteria part as text; absent/None/empty-after-stringify -> fallback."""
    if value is None:
        return fallback
    text = serialize_text(value)
    return text if text.strip() else fallback


def _options_for(question) -> list[dict]:
    if isinstance(question, NoulQuestion):
        return [
            {"id": "true",
             "description": _rubric_or(
                 question.criteria.true if question.criteria else None, "Yes")},
            {"id": "false",
             "description": _rubric_or(
                 question.criteria.false if question.criteria else None, "No")},
        ]
    if isinstance(question, ChoiceQuestion):
        return [{"id": option,
                 "description": rubric if rubric is not None
                 else _NO_ADDITIONAL_DESCRIPTION}
                for option, rubric in question.criteria.items()]
    if isinstance(question, ScoreQuestion):
        return [{"id": str(i), "description": serialize_text(level)}
                for i, level in enumerate(question.criteria)]
    raise SemIfValidationError("Unsupported question type")  # union prevents this


def build_semif_rows(request: SystemOneRequest) -> list[dict]:
    rows = [{"id": question_id,
             "state": request.state,                     # verbatim str/dict/list
             "question": serialize_text(question.instructions),
             "options": _options_for(question)}
            for question_id, question in request.questions.items()]
    for row in rows:
        try:
            validate_row(row)
        except ValueError as exc:
            raise SemIfValidationError(f"Question {row['id']!r}: {exc}") from exc
    return rows