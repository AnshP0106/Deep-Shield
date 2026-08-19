from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.services.upload_service import process_uploaded_media
from app.services.analysis_service import analyze_media


router = APIRouter()


@router.post("/upload")
def upload_media(file: UploadFile = File(...)):

    upload_result = process_uploaded_media(file)

    analysis_result = analyze_media(
        file_path=Path(upload_result.saved_to),
        media_type=upload_result.media_type
    )

    return {
        "upload": upload_result,
        "analysis": analysis_result
    }