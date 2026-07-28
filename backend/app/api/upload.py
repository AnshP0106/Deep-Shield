from pathlib import Path
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException 

#Create router
router = APIRouter()

from app.core.config import (
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    SUPPORTED_AUDIO_TYPES,
    UPLOAD_DIRECTORIES
)

@router.get("/upload/test")
def test_upload_router():
    return {
        "message": "Upload router is working successfully!"
    }


@router.post("/upload")
def upload_media(file: UploadFile = File(...)):

    # Detect media type
    if file.content_type in SUPPORTED_IMAGE_TYPES:
        media_type = "image"

    elif file.content_type in SUPPORTED_VIDEO_TYPES:
        media_type = "video"

    elif file.content_type in SUPPORTED_AUDIO_TYPES:
        media_type = "audio"

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported media type."
        )

    UPLOAD_DIRECTORIES = {
    "image": Path("uploads/images"),
    "video": Path("uploads/videos"),
    "audio": Path("uploads/audio"),
}

    destination_folder = UPLOAD_DIRECTORIES[media_type]

    # Create complete file path
    file_path = destination_folder / file.filename

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully!",
        "filename": file.filename,
        "media_type": media_type,
        "saved_to": str(file_path)
    }
