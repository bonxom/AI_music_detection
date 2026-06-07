from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.route.ai import router as ai_router

app = FastAPI(title="AI Music Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(ai_router)
