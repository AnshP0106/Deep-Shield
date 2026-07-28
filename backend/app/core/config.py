from pathlib import Path

#Supported MIME Types
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff"
}

SUPPORTED_VIDEO_TYPES = {
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska"
}

SUPPORTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/aac",
    "audio/mp4",
    "audio/ogg"
}

#Upload Directories
UPLOAD_DIRECTORIES = {
    "image": Path("uploads/images"),
    "video": Path("uploads/videos"),
    "audio": Path("uploads/audio")
}