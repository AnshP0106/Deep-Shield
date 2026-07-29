from pathlib import Path
import shutil
import uuid

from fastapi import UploadFile, HTTPException

from app.core.config import (
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    SUPPORTED_AUDIO_TYPES,
    UPLOAD_DIRECTORIES
)


def process_uploaded_media(file: UploadFile):

    # -----------------------------
    # Detect media type
    # -----------------------------

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

    # -----------------------------
    # Get upload directory
    # -----------------------------

    destination_folder = UPLOAD_DIRECTORIES[media_type]

    # -----------------------------
    # Generate unique filename
    # -----------------------------

    extension = Path(file.filename).suffix

    unique_filename = f"{uuid.uuid4()}{extension}"

    file_path = destination_folder / unique_filename

    # -----------------------------
    # Save file
    # -----------------------------

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -----------------------------
    # Return response
    # -----------------------------

    return {
        "message": "File uploaded successfully!",
        "original_filename": file.filename,
        "stored_filename": unique_filename,
        "content_type": file.content_type,
        "media_type": media_type,
        "saved_to": str(file_path)
    }