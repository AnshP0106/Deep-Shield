from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.upload import router
from app.managers.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting DeepShield...")

    model_manager.load_models()

    yield

    print("Shutting down DeepShield...")


app = FastAPI(
    title="DeepShield API",
    lifespan=lifespan
)

# =========================================================
# SERVE GENERATED OUTPUT FILES
# =========================================================
#
# This allows FastAPI to serve:
#
# outputs/gradcam/face_1_gradcam.jpg
#
# through:
#
# http://127.0.0.1:8000/outputs/gradcam/face_1_gradcam.jpg
#
# =========================================================

app.mount(
    "/outputs",
    StaticFiles(directory="outputs"),
    name="outputs"
)


# =========================================================
# API ROUTES
# =========================================================

app.include_router(router)