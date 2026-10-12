from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.upload import router
from app.managers.model_manager import model_manager


# =========================================================
# PATHS
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BACKEND_DIR / "outputs"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"

# Ensure output directories exist before mounting them.
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# APPLICATION LIFECYCLE
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting DeepShield...")

    model_manager.load_models()

    yield

    print("Shutting down DeepShield...")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="DeepShield API",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =========================================================
# GENERATED EVIDENCE FILES
# =========================================================

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs",
)


# =========================================================
# API ROUTES
# =========================================================

app.include_router(router)