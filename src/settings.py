import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent  # .../semif-api-jev-schema


def _empty_to_none(v):
    return None if isinstance(v, str) and v.strip() == "" else v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # auth (unchanged)
    API_KEY: str | None = None
    USE_API_KEY: bool = False
    HF_HOME: str | None = None

    # SemIf backend
    SEMIF_MODEL: str = ""            # empty -> startup aborts with a clear message
    SEMIF_REVISION: str = ""         # empty -> startup aborts (load_model requires it)
    SEMIF_BACKEND: str = "mlx"       # "mlx" only in this revision (torch needs CUDA)
    SEMIF_MODE: str = "direct"       # "direct" (parity-pinned) | "serial" (state prefix
                                     # cache; ~2.6x faster multi-question; documented drift)
    SEMIF_MAX_TOKENS: int = 4096     # prompt token limit -> 422 beyond
    SEMIF_MLX_BITS: int | None = None            # in-memory 4/8-bit quantization
    SEMIF_MLX_CACHE_LIMIT_MIB: int | None = None # None -> mlx_backend default 256
    SEMIF_PRELOAD: bool = True       # load at startup (lifespan) vs first request

    # Jev response surface
    REPORTED_MODEL_ID: str = ""      # empty -> echo the client's model string
    ALLOWED_MODELS: str = ""         # comma list; empty -> accept any

    @field_validator("SEMIF_MLX_BITS", "SEMIF_MLX_CACHE_LIMIT_MIB", mode="before")
    @classmethod
    def _coerce_empty(cls, v):
        return _empty_to_none(v)     # "" in .env means "unset", not an error


config = Settings()

if config.HF_HOME:
    os.environ.setdefault("HF_HOME", config.HF_HOME)  # exported env vars still win