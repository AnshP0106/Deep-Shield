from fastapi import APIRouter, UploadFile, File

from app.services.upload_service import process_uploaded_media
from app.schemas.upload_schema import UploadResponse

router = APIRouter()


@router.get("/upload/test")
def test_upload_router():
    return {
        "message": "Upload router is working successfully!"
    }


@router.post("/upload", response_model=UploadResponse)
def upload_media(file: UploadFile = File(...)):
    return process_uploaded_media(file)