"""SemIf result rows -> SystemOneResponse (plan-rev-1 §4 D4/D5).

SemIf returns probabilities aligned with option_ids in declared order, so the
old sorted-by-probability telemetry hazard is gone. The probability math is
identical to the previous revision and pinned by the same (ported) tests.
"""

from classes.jev_errors import SemIfRuntimeError
from classes.jev_translator import serialize_text
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
from settings import config


# --------------------------------------------------------------------------
# Probability math (unchanged from the previous revision; zero-mass now
# raises SemIfRuntimeError -> 500 instead of SourceEngineError -> 502)
# --------------------------------------------------------------------------


def _round4(value: float) -> float:
    return round(value, 4)


def _renormalize(probabilities: dict[str, float], question_id: str) -> dict[str, float]:
    """Round to 4 decimals, then renormalize so the map sums to exactly 1 by
    adding the residual to the argmax option (Jev documents "floats that sum
    to 1"). Ties on the argmax resolve to the first declared key.
    """
    rounded = {key: _round4(value) for key, value in probabilities.items()}
    total = sum(rounded.values())
    if total <= 0.0:
        raise SemIfRuntimeError(
            f"Scorer probabilities for {question_id!r} carry no usable mass"
        )
    residual = _round4(1.0 - total)
    if residual != 0.0:
        argmax = max(rounded, key=rounded.get)
        rounded[argmax] = _round4(rounded[argmax] + residual)
    return rounded


def _confidence(probabilities: dict[str, float]) -> float:
    """TypeSafe's published confidence definition (docs.typesafe.ai/confidence):
    for n options with peak probability p_max,

        confidence = clamp((n x p_max - 1) / (n - 1), 0, 1)        # n >= 2

    All mass on one option -> 1.0; uniform spread -> 0.0. This is computed
    from the (renormalized) distribution.
    """
    n = len(probabilities)
    if n < 2:
        return 1.0
    p_max = max(probabilities.values())
    value = (n * p_max - 1) / (n - 1)
    return _round4(min(1.0, max(0.0, value)))


# --------------------------------------------------------------------------
# SemIf result access
# --------------------------------------------------------------------------


def _distribution(result: dict, option_ids: list[str], question_id: str) -> dict[str, float]:
    """declared option id -> probability (missing -> 0.0, defensive)."""
    got = result.get("option_ids")
    probs = result.get("probabilities")
    if not isinstance(got, list) or not isinstance(probs, list) or len(got) != len(probs):
        raise SemIfRuntimeError(
            f"Scorer result for {question_id!r} has misaligned option_ids/probabilities")
    if got != option_ids:
        raise SemIfRuntimeError(
            f"Scorer result for {question_id!r} does not match declared options")
    table = dict(zip(got, (float(p) for p in probs)))
    return {option: table.get(option, 0.0) for option in option_ids}


def _argmax(distribution: dict[str, float]) -> str:
    """First declared key holding the max value (dict preserves declared order)."""
    return max(distribution, key=distribution.get)


# --------------------------------------------------------------------------
# Per-question answer builders
# --------------------------------------------------------------------------


def _noul_answer(question_id, result) -> NoulAnswer:
    distribution = _distribution(result, ["true", "false"], question_id)
    return NoulAnswer(type="noul", noul=_round4(distribution["true"]))
    # no renormalization, no confidence — same as the previous revision


def _choice_answer(question_id, question: ChoiceQuestion, result) -> ChoiceAnswer:
    declared = list(question.criteria.keys())
    distribution = _renormalize(
        _distribution(result, declared, question_id), question_id)
    return ChoiceAnswer(type="choice", choice=_argmax(distribution),
                        probabilities=distribution, confidence=_confidence(distribution))


def _score_answer(question_id, question: ScoreQuestion, result) -> ScoreAnswer:
    declared = [str(i) for i in range(len(question.criteria))]
    distribution = _renormalize(
        _distribution(result, declared, question_id), question_id)
    score = _round4(sum(i * distribution[str(i)] for i in range(len(question.criteria))))
    legend = {str(i): serialize_text(level) for i, level in enumerate(question.criteria)}
    return ScoreAnswer(type="score", score=score, legend=legend,
                       probabilities=distribution, confidence=_confidence(distribution))


# --------------------------------------------------------------------------
# Top-level builder
# --------------------------------------------------------------------------


def build_systemone_response(
    request: SystemOneRequest, rows: list[dict], results: list[dict]
) -> SystemOneResponse:
    """Assemble the Jev response from the scored SemIf rows.

    Raises SemIfRuntimeError (-> 500) on any result/row misalignment — the
    proxy never invents answers.
    """
    if len(results) != len(rows):
        raise SemIfRuntimeError("Scorer returned the wrong number of result rows")
    by_id = {row["id"]: result for row, result in zip(rows, results)}
    answers = {}
    for question_id, question in request.questions.items():
        result = by_id.get(question_id)
        if result is None or result.get("id") != question_id:
            raise SemIfRuntimeError(f"Scorer result missing for question {question_id!r}")
        if isinstance(question, NoulQuestion):
            answers[question_id] = _noul_answer(question_id, result)
        elif isinstance(question, ChoiceQuestion):
            answers[question_id] = _choice_answer(question_id, question, result)
        else:
            answers[question_id] = _score_answer(question_id, question, result)
    usage = Usage(
        input_tokens=sum(int(result["input_tokens"]) for result in results),  # real counts
        output_tokens=0,                                                      # no generated tokens
    )
    # model echo: REPORTED_MODEL_ID if set, else the model the client sent.
    model = config.REPORTED_MODEL_ID or request.model
    return SystemOneResponse(model=model, answers=answers, usage=usage)


# --------------------------------------------------------------------------
# Timing summary (Fix 4, implementation-rev-2 §4.1) — response headers only,
# the Jev response schema is untouched
# --------------------------------------------------------------------------


def scoring_summary(results: list[dict]) -> dict | None:
    """Per-request timing summary from SemIf result rows, or None when the
    rows carry no timing (e.g. test fakes). Row totals sum encode+forward
    per row; queue wait is NOT included (headers carry the handler wall time)."""
    totals = [r["total_seconds"] for r in results
              if isinstance(r.get("total_seconds"), (int, float))]
    forwards = [r["forward_seconds"] for r in results
                if isinstance(r.get("forward_seconds"), (int, float))]
    if not totals:
        return None
    serving = results[0].get("model")
    return {
        "scoring_ms": round(sum(totals) * 1000),
        "forward_ms": round(sum(forwards) * 1000) if forwards else None,
        "rows": len(results),
        "mode": serving.get("serving_config") if isinstance(serving, dict) else None,
    }