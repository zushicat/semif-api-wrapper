"""Jev-compatible routes (plan-rev-1 §4 D1/D3).

- ``POST /v1/systemone``: translates the request into SemIf decision rows and
  scores them in-process; returns the Jev response shape.
- ``GET /v1/models``: static local catalog — not required for basic drop-in
  use, but it makes the TypeSafe SDK (``client.models.list()``) work out of
  the box.

Error semantics: pydantic validation failures surface as FastAPI's default
422; SemIfValidationError (row validation, over-limit prompt) maps to 422 and
SemIfRuntimeError (scoring failed) maps to 500, always with FastAPI's
``{"detail": ...}`` body. A client that disconnects before/during scoring is
answered with a bare 499 (uvicorn discards it — nobody is listening); the
scorer skips the remaining rows of an abandoned request.

Timing visibility: successful responses carry informational ``X-Request-Ms`` /
``X-Scoring-*`` headers plus one ``scored rows=...`` log line. They are not
part of the Jev contract (the response schema and OpenAPI are unchanged).
"""

import asyncio
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Request, Response

from classes.jev_errors import SemIfCancelled, SemIfRuntimeError, SemIfValidationError
from classes.jev_response_builder import build_systemone_response, scoring_summary
from classes.jev_translator import build_semif_rows
from models.jev_models import SystemOneRequest, SystemOneResponse
from semif_runtime import runtime
from settings import config

# logger = logging.getLogger(__name__)

router = APIRouter()

#: Date this local Jev imitation was implemented; reported as ``release_date``
#: in the model catalog.
IMPLEMENTATION_DATE = "2026-09-20"

_MODEL_CATALOG = [
    {
        "name": "jev-latest",
        "description": (
            "Jev-compatible proxy backed by the in-process SemIf scorer "
            "(MLX direct mode, native option logits)."
        ),
        "release_date": IMPLEMENTATION_DATE,
    },
    {
        "name": "jev-preview",
        "description": (
            "Pre-release alias served by the same in-process SemIf scorer; "
            "identical behavior to jev-latest."
        ),
        "release_date": IMPLEMENTATION_DATE,
    },
]


@router.get("/v1/models")
async def list_models():
    return {"models": _MODEL_CATALOG}


def _validate_model(model: str) -> None:
    """Model is validated as a non-empty string by the request model and
    otherwise ignored. With ``ALLOWED_MODELS`` set (comma list), only listed
    ids are accepted; empty means accept any model string."""
    allowed_raw = (config.ALLOWED_MODELS or "").strip()
    if not allowed_raw:
        return
    allowed = [entry.strip() for entry in allowed_raw.split(",") if entry.strip()]
    if model not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"model {model!r} is not allowed (ALLOWED_MODELS={allowed_raw!r})",
        )


#: Disconnect poll interval (analysis-rev-2 §1.5: disconnects are observed
#: within ~30-60 ms of the TCP close on this stack; 50 ms is plenty).
_CANCEL_POLL_S = 0.05


async def _watch_disconnect(raw: Request, cancelled: threading.Event) -> None:
    """Set `cancelled` as soon as the client goes away."""
    while not cancelled.is_set():
        if await raw.is_disconnected():
            cancelled.set()
            return
        await asyncio.sleep(_CANCEL_POLL_S)


@router.post("/v1/systemone", response_model=SystemOneResponse)
async def systemone(
    request: SystemOneRequest, raw: Request, response: Response
) -> SystemOneResponse | Response:
    t0 = time.perf_counter()
    _validate_model(request.model)
    if await raw.is_disconnected():  # cheap early-out: client already gone
        return Response(status_code=499)
    cancelled = threading.Event()
    watcher = asyncio.create_task(_watch_disconnect(raw, cancelled))
    try:
        rows = build_semif_rows(request)
        results = await asyncio.to_thread(runtime.score_rows, rows, cancelled)
        body = build_systemone_response(request, rows, results)
        _set_timing_headers(response, results, t0)
        return body
    except SemIfCancelled as exc:
        # logger.info("request abandoned by client: %s", exc)
        return Response(status_code=499)  # client is gone; uvicorn discards it
    except SemIfValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SemIfRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        watcher.cancel()  # never leak the poller task


def _set_timing_headers(response: Response, results: list[dict], t0: float) -> None:
    """Attach the informational timing headers and log one summary line.

    `X-Request-Ms` is the handler wall time (includes queue wait behind other
    requests' rows); `X-Scoring-Ms` is Σ row totals, so `X-Request-Ms −
    X-Scoring-Ms ≈ wait`. Rows without timing fields (test fakes) set nothing.
    Cancelled requests never reach this — nobody is listening anyway."""
    summary = scoring_summary(results)
    wall_ms = round((time.perf_counter() - t0) * 1000)
    if summary is not None:
        response.headers["X-Request-Ms"] = str(
            wall_ms
        )  # handler wall time incl. queue wait
        response.headers["X-Scoring-Ms"] = str(summary["scoring_ms"])  # Σ row totals
        response.headers["X-Scoring-Rows"] = str(summary["rows"])
        if summary["forward_ms"] is not None:
            response.headers["X-Scoring-Forward-Ms"] = str(summary["forward_ms"])
        if summary["mode"]:
            response.headers["X-Scoring-Mode"] = str(summary["mode"])
    # logger.info(
    #     "scored rows=%s mode=%s wall_ms=%s scoring_ms=%s forward_ms=%s",
    #     summary["rows"] if summary else "?",
    #     summary["mode"] if summary else "?",
    #     wall_ms,
    #     summary["scoring_ms"] if summary else "?",
    #     summary["forward_ms"] if summary else "?",
    # )
