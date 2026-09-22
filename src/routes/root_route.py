from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root_get():
    return {"Status": "ok", "Response": "GET Root"}


@router.post("/")
async def root_post():
    return {"Status": "ok", "Response": "POST Root"}
