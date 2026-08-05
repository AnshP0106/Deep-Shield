from pathlib import Path

from app.schemas.analysis_schema import AnalysisResponse


def analyze_video(file_path: Path) -> AnalysisResponse:
    """
    Placeholder for the Video Deepfake Detection Model.
    """

    return AnalysisResponse(
        status="Pending",
        media_type="video",
        prediction=None,
        confidence=None,
        model_name=None,
        explanation="Video AI model not integrated yet."
    )