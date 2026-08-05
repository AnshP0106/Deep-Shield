from contextlib import asynccontextmanager

from fastapi import FastAPI

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

app.include_router(router)