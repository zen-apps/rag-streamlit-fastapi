from fastapi import (
    FastAPI,
)
import sys

sys.path.append("fast_api")

from app.api.genai import genai

app = FastAPI(
    title="AI Chat",
    version=1.0,
    description="AI Chat",
)

app.include_router(
    genai,
    prefix="/v1/genai",
    tags=["genai"],
)
