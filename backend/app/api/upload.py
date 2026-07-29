from fastapi import APIRouter, UploadFile, File

from app.services.upload_service import process_uploaded_media

router = APIRouter()


@router.get("/upload/test")
def test_upload_router():
    return {
        "message": "Upload router is working successfully!"
    }


@router.post("/upload")
def upload_media(file: UploadFile = File(...)):
    return process_uploaded_media(file)