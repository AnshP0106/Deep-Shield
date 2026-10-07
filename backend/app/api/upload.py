from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from app.services.upload_service import process_uploaded_media
from app.services.analysis_service import analyze_media
from app.services.explanation_service import generate_explanation


router = APIRouter()


@router.post("/upload")
def upload_media(file: UploadFile = File(...)):

    # -------------------------------------------------
    # Upload media
    # -------------------------------------------------

    upload_result = process_uploaded_media(file)

    # -------------------------------------------------
    # Analyze media
    # -------------------------------------------------

    analysis_result = analyze_media(
        file_path=Path(upload_result.saved_to),
        media_type=upload_result.media_type
    )

    # -------------------------------------------------
    # Generate AI explanation
    # -------------------------------------------------

    explanation = generate_explanation(
        media_type=upload_result.media_type,
        detection_result=analysis_result.model_dump()
    )

    # -------------------------------------------------
    # Final response
    # -------------------------------------------------

    return {
        "upload": upload_result,
        "analysis": analysis_result,
        "explanation": explanation
    }