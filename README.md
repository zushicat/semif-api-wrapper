# Semif Api Wrapper
*IMPORTANT*    
This is an Api wrapper for [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) using [Typesafe Ai's Jev Api schema](https://docs.typesafe.ai/api).
The code in [/src/semif_phase1/](/src/semif_phase1/) is the Python code from the original SemIf project (as of Sept. 22, 2026).

#
A local, Jev-compatible decision API: clients send a **Jev request** and get a
**Jev response** back, but scoring runs **in-process** with the SemIf scorer
(`src/semif_phase1`) on Apple-Silicon MLX — no remote engine, no generated
tokens. Probabilities come from native option logits on one forward pass per
question.

## Request mapping (Jev question → SemIf row)

One row per question, in `questions` insertion order; `state` passes through
verbatim (str / dict / list — no stringification), `question` =
`serialize_text(instructions)`:

| Jev question | SemIf `options` |
|---|---|
| `noul` | `[{"id": "true", "description": criteria.true or "Yes"}, {"id": "false", "description": criteria.false or "No"}]` |
| `choice` | `[{"id": key, "description": rubric or "(no additional description)"} for key, rubric in criteria.items()]` — insertion order preserved |
| `score` | `[{"id": str(i), "description": serialize_text(level)} for i, level in enumerate(criteria)]` |

## Response mapping (SemIf result row → Jev answer)

SemIf returns probabilities aligned with `option_ids` (declared order), so the
old sorted-by-probability telemetry hazard is gone:

| Jev answer | Built from the result row |
|---|---|
| `noul` | `noul` = probability of option `"true"` (round 4); no confidence |
| `choice` | `probabilities` = declared key → probability (round 4, renormalized so the map sums to exactly 1, residual → argmax); `choice` = argmax (ties → first declared key); `confidence` = `clamp((n·p_max−1)/(n−1), 0, 1)` |
| `score` | `probabilities` over `"0".."N-1"` (same rounding); `score` = `Σ i·pᵢ` (may land between levels); `legend` = request criteria verbatim; `confidence` = same formula |

`usage.input_tokens` = real sum of the rows' token counts (no chars/4
estimate); `usage.output_tokens` = 0 (option logits, no generated tokens).
`model` = `REPORTED_MODEL_ID` if set, else the client's model string.

## Documented deviations from Jev

| | Jev | This API |
|---|---|---|
| Choice options | 1–255 | **2–16** (SemIf option bounds) → 422 outside |
| Prompt tokens | — | hard limit `SEMIF_MAX_TOKENS` (default 4096), no truncation → 422 beyond |
| Engine unreachable | 502 | **gone** (no remote hop) |
| Scoring failure | 502 | **500** — the proxy never invents answers |
| Missing model at startup | lazy load | **process refuses to start**, settings echoed |
| Serial-mode drift | — | `SEMIF_MODE=serial` argmax drift vs `direct`: 5/777 upstream-measured, 0/3 on the local fixtures (2/3 rows differ in late decimals only); serial golden pinned at `docs/Semif/results-mlx-serial.jsonl`, numbers in `plans/rev-2/serial-drift-report.txt` |

Auth (401 missing header / 401 bad scheme / 403 wrong token) and pydantic
422s are unchanged.

## Configuration (`.env` — see `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `SEMIF_MODEL` | *(required)* | Model path or HF id; missing → startup abort naming the variable |
| `SEMIF_REVISION` | *(required)* | Revision label (required by `load_model` for local paths too) |
| `SEMIF_BACKEND` | `mlx` | `mlx` only in this revision (torch needs CUDA) |
| `SEMIF_MODE` | `direct` | `direct` = parity-pinned golden, one forward pass per row; `serial` = state prefix cache, ≈2.6× faster on multi-question requests (documented drift — see deviations); `shared` still rejected ("not implemented yet") |
| `SEMIF_MAX_TOKENS` | `4096` | Prompt token limit → 422 beyond |
| `SEMIF_MLX_BITS` | *(empty)* | Optional in-memory 4/8-bit quantization |
| `SEMIF_MLX_CACHE_LIMIT_MIB` | *(empty = 256)* | MLX allocator cache (MiB) |
| `SEMIF_PRELOAD` | `true` | Load at startup (lifespan) vs. first request |
| `REPORTED_MODEL_ID` | *(empty → echo)* | Override the reported model id |
| `ALLOWED_MODELS` | *(empty → accept any)* | Comma list of accepted model ids |
| `API_KEY` / `USE_API_KEY` / `HF_HOME` | — | Unchanged |

Note: `HF_HOME` only matters when models resolve through the HF hub cache; a
local `SEMIF_MODEL` path loads directly. `.env` changes need a manual server
restart — `--reload` watches only `.py` files.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[mlx,test]'     # mlx extra on Apple Silicon; test extra for pytest-asyncio
uvicorn app:app --reload --host 0.0.0.0 --port 11100   # from src/: uvicorn app:app ...
```

With `SEMIF_PRELOAD=true` the model loads at startup (tens of seconds; a
missing `SEMIF_MODEL`/`SEMIF_REVISION` aborts startup). With `--reload`, every
`.py` save restarts the worker and re-loads the model — expected. Switching
`SEMIF_MODE` (or any `.env` value) always needs a manual restart.

Optional overload guardrail (rev-2, deliberately **not** the default run
command): `uvicorn app:app --host 0.0.0.0 --port 11100 --limit-concurrency 8`
answers excess requests with 503 immediately instead of queueing them
unboundedly; the game maps ≥ 500 to its fallback, so callers fail fast instead
of hanging behind a full GPU queue.

## Test

```bash
pytest -q                 # unit suite (143 tests); integration deselected
pytest -q -m integration  # golden live tests: replay docs/Semif/decisions.jsonl through
                          # the real model — direct mode vs results-mlx-direct.jsonl and
                          # serial mode vs results-mlx-serial.jsonl, 4-decimal parity
                          # (skips without the model)
```

Quick live check:

```bash
curl -s -X POST http://localhost:11100/v1/systemone \
  -H 'Content-Type: application/json' -d '{
    "state": "My cat has food allergies. Does your pet food contain any allergens?",
    "model": "jev-latest",
    "questions": {
      "is_request": { "type": "noul", "instructions": "Is this a request?" },
      "department": { "type": "choice", "instructions": "What kind of pet does the customer have?",
                      "criteria": { "CAT": null, "BIRD": null, "DOG": null } }
    }
  }' | python3 -m json.tool
```

`test_request.http` (VS Code REST Client) has the full example set.

## Architecture

```
client --Jev--> POST /v1/systemone
                   │  src/routes/jev_route.py
                   │  1. build_semif_rows(request)           → N decision rows
                   │  2. runtime.score_rows(rows, cancelled) → N result rows
                   │     (asyncio.to_thread; the module lock is cycled per row;
                   │      a disconnect watcher sets `cancelled`, so abandoned
                   │      requests skip their remaining rows)
                   │  3. build_systemone_response(...)       → Jev response
                   ▼
             model + answers + usage  (+ informational X-Request-Ms / X-Scoring-* headers)
```

- `src/semif_runtime.py` — module singleton. `load()` is idempotent and called
  from the FastAPI lifespan (`SEMIF_PRELOAD=true`, fail fast at startup).
  Scoring is serialized **per row** behind a `threading.Lock` (MLX/Metal is not
  thread-safe) while the event loop stays free via `asyncio.to_thread`: a row
  in flight always finishes, but the cancel flag is checked before every row,
  so an abandoned request wastes at most the one row already in flight (a
  Python thread cannot be killed mid-`mx.eval`). Lock waits are bounded by one
  row (~0.5 s for the game payload) instead of one whole request — still not
  FIFO (`threading.Lock` gives no fairness guarantee).
- Disconnect cancellation (rev-2 Fix 1): `src/routes/jev_route.py` polls
  `request.is_disconnected()` every 50 ms (`_watch_disconnect`) and sets the
  cancel event the moment the client goes away; the route answers a bare
  **499** (uvicorn discards it — nobody is listening) and logs
  `request abandoned by client: ... N/M rows`.
- Serial mode (rev-2 Fix 2): with `SEMIF_MODE=serial` the runtime creates one
  stateful `SerialPrefixScorer` **per request** — all rows of a Jev request
  share the same state, so rows 2..N hit the exact-prefix cache instead of
  re-prefilling (≈2.6× faster on the game payload; per-request instance, so
  nothing leaks across requests). Drift vs `direct` is documented below.
- Timing visibility (rev-2 Fix 4): every successful response carries
  informational `X-Request-Ms` (handler wall time incl. queue wait),
  `X-Scoring-Ms` (Σ row totals), `X-Scoring-Rows`, `X-Scoring-Forward-Ms` and
  `X-Scoring-Mode` headers — `X-Request-Ms − X-Scoring-Ms ≈ wait behind other
  requests' rows` — plus one `scored rows=... mode=... wall_ms=...` log line
  per request.
- `src/classes/jev_translator.py` — one SemIf row per question
  (`{id, state, question, options}`), validated with SemIf's own
  `validate_row` so row-shape problems 422 before any model work.
- `src/classes/jev_response_builder.py` — result rows → Jev answers; the
  probability math (round 4, renormalize-to-1, confidence formula, score
  expectation) is unchanged from the previous engine-based revision.

## Limitations

- Scoring is serialized **per row** behind one lock — MLX/Metal is not
  thread-safe; concurrent requests queue with waits bounded by one row
  (~0.5 s for the game payload), but lock handoff is **not FIFO**.
- A row already in flight when the client disconnects still completes (threads
  cannot be killed mid-`mx.eval`); every later row of that request is skipped.
- `serial` mode is **not bit-identical** to `direct` (argmax drift, see
  deviations) — `direct` stays the default; pin your own golden before
  trusting serial numbers. `SEMIF_MODE=shared` (one prefill + parallel
  suffixes) is still rejected ("not implemented yet").
- `X-Request-Ms` / `X-Scoring-*` headers and the `scored rows=...` log line are
  informational — not part of the Jev contract (response schema and OpenAPI
  unchanged).
- The model still loads from the external drive `/Volumes/Expansion` (2.9 GB
  over USB; ≈44 s cold, faster when cached) — moving it to the internal disk
  (rev-2 analysis Fix 3) is explicitly **out of scope** for this revision.
- Choice questions accept 2–16 options (not Jev's published 1–255); score
  questions keep 2–10 levels.
- Prompts are never truncated; oversized prompts are a 422.
- Answer-slot tokens (`A`–`P`) must be exact round-trip tokens — verified per
  request by SemIf; a tokenizer change would 500 loudly, never silently.
- `semif_phase1` still imports the transformers/torch stack even in mlx mode
  (stage-1 accepted state; import trimming is a later optimization).
- Probabilities are conditional option scores — uncalibrated as decision
  confidence (SemIf semantics).