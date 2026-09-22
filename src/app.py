import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth import bearer_auth_dependency
from routes import jev_route, root_route
from semif_runtime import runtime
from settings import config

# Uvicorn only configures its own uvicorn.* loggers; without this the route's
# INFO lines ("scored rows=...", "request abandoned by client: ...") inherit
# the root logger's WARNING level and are silently dropped.
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if config.SEMIF_PRELOAD:
        runtime.load()        # blocks startup until the model is ready (fail fast)
    yield
    runtime.reset()


app = FastAPI(dependencies=[Depends(bearer_auth_dependency)], lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # local dev proxy; the game calls it from http://localhost:6999
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
app.include_router(root_route.router)
app.include_router(jev_route.router)