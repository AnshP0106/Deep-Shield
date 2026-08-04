from pathlib import Path

from app.schemas.analysis_schema import AnalysisResponse


def analyze_media(file_path: Path, media_type: str) -> AnalysisResponse:

    return AnalysisResponse(
        status="Pending",
        media_type=media_type,
        prediction=None,
        confidence=None,
        model_name=None,
        explanation="AI model not integrated yet."
    )