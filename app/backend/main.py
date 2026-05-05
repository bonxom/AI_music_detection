from fastapi import FastAPI

from app.backend.route.ai import router as ai_router

app = FastAPI(title="AI Music Detection API")
app.include_router(ai_router)
