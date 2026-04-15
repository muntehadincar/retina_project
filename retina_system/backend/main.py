"""
main.py — FastAPI giriş noktası.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from services import model_service
from routers import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()
    yield

app = FastAPI(
    title="Retina Vessel Segmentation API",
    description="ResUNet ile retinal damar segmentasyonu",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "public")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
