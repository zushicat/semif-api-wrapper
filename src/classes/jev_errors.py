class SemIfValidationError(Exception):
    """The request cannot be served as translated: row validation failed,
    prompt exceeds SEMIF_MAX_TOKENS, or bounds violated. Route maps to 422."""


class SemIfRuntimeError(Exception):
    """In-process scoring failed for a non-client reason (slot-token
    collision, model error, malformed result). Route maps to 500."""


class SemIfCancelled(Exception):
    """The client disconnected before/during scoring; the route answers 499
    and uvicorn discards it (nobody is listening). Not a scoring failure and
    never a 500. Carries progress for the log line."""

    def __init__(self, rows_scored: int, rows_total: int):
        super().__init__(f"client disconnected after {rows_scored}/{rows_total} rows")
        self.rows_scored = rows_scored
        self.rows_total = rows_total