from pathlib import Path

from app.schemas.analysis_schema import AnalysisResponse
from app.detectors.image_detector import image_detector


def analyze_media(
    file_path: Path,
    media_type: str
) -> AnalysisResponse:

    if media_type == "image":

        result = image_detector.predict(file_path)

        return AnalysisResponse(
            media_type=media_type,
            result=result["prediction"],
            confidence=result["confidence"],
            details={
                "real_probability": result["real_probability"],
                "fake_probability": result["fake_probability"],
                "face_detected": result["face_detected"]
            }
        )

    return AnalysisResponse(
        media_type=media_type,
        result="NOT_ANALYZED",
        confidence=0.0,
        details={
            "message": f"{media_type} analysis is not implemented yet."
        }
    )