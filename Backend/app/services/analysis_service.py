from pathlib import Path

from app.detectors.audio_detector import analyze_audio
from app.detectors.image_detector import analyze_image
from app.detectors.video_detector import analyze_video
from app.schemas.analysis_schema import AnalysisResponse


def analyze_media(file_path: Path, media_type: str) -> AnalysisResponse:
    """
    Routes uploaded media to the correct AI detector.
    """

    if media_type == "image":
        return analyze_image(file_path)

    if media_type == "video":
        return analyze_video(file_path)

    if media_type == "audio":
        return analyze_audio(file_path)

    return AnalysisResponse(
        status="Failed",
        media_type=media_type,
        explanation="Unsupported media type."
    )