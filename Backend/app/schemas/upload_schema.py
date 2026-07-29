from typing import Optional

from pydantic import BaseModel


class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    mode: str


class UploadResponse(BaseModel):
    message: str
    original_filename: str
    stored_filename: str
    content_type: str
    media_type: str
    saved_to: str
    metadata: Optional[ImageMetadata] = None