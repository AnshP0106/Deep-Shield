from pathlib import Path
from app.schemas.upload_schema import UploadResponse, ImageMetadata
import shutil
import uuid
from app.services.metadata_service import extract_image_metadata
from fastapi import UploadFile


from app.core.config import (
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    SUPPORTED_AUDIO_TYPES,
    UPLOAD_DIRECTORIES
)

from app.exceptions.upload_exceptions import UnsupportedMediaTypeException


def detect_media_type(content_type: str) -> str:
    if content_type in SUPPORTED_IMAGE_TYPES:
        return "image"

    if content_type in SUPPORTED_VIDEO_TYPES:
        return "video"

    if content_type in SUPPORTED_AUDIO_TYPES:
        return "audio"

    raise UnsupportedMediaTypeException()


def generate_unique_filename(original_filename: str) -> str:
    extension = Path(original_filename).suffix
    return f"{uuid.uuid4()}{extension}"


def save_uploaded_file(file: UploadFile, destination: Path) -> None:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def process_uploaded_media(file: UploadFile) -> UploadResponse:
    media_type = detect_media_type(file.content_type)

    unique_filename = generate_unique_filename(file.filename)

    destination_folder = UPLOAD_DIRECTORIES[media_type]

    file_path = destination_folder / unique_filename

    save_uploaded_file(file, file_path)

    # Extract metadata only for images
    metadata = {}

    if media_type == "image":
        metadata = extract_image_metadata(str(file_path))

    image_metadata = None

    if media_type == "image":
        image_metadata = ImageMetadata(**metadata)

    return UploadResponse(
        message="File uploaded successfully!",
        original_filename=file.filename,
        stored_filename=unique_filename,
        content_type=file.content_type,
        media_type=media_type,
        saved_to=str(file_path),
        metadata=image_metadata
    )