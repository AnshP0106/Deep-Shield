from pathlib import Path

from app.schemas.analysis_schema import AnalysisResponse
from app.detectors.image_detector import image_detector


def analyze_media(
    file_path: Path,
    media_type: str
) -> AnalysisResponse:

    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    if media_type == "image":

        # Run multi-face analysis
        result = image_detector.predict_multiple_faces(
            file_path
        )

        # -------------------------------------------------
        # Return analysis response
        # -------------------------------------------------

        return AnalysisResponse(
            media_type=media_type,

            result=result["prediction"],

            confidence=result["confidence"],

            details={
                "face_count": result["face_count"],

                "fake_face_count": result.get(
                    "fake_face_count",
                    0
                ),

                "real_face_count": result.get(
                    "real_face_count",
                    0
                ),

                "uncertain_face_count": result.get(
                    "uncertain_face_count",
                    0
                ),

                "risk_score": result.get(
                    "risk_score",
                    0.0
                ),

                "risk_level": result.get(
                    "risk_level",
                    "UNKNOWN"
                ),

                "risk_reason": result.get(
                    "risk_reason",
                    ""
                ),

                "faces": result.get(
                    "faces",
                    []
                )
            }
        )

    # =====================================================
    # OTHER MEDIA TYPES
    # =====================================================

    return AnalysisResponse(
        media_type=media_type,

        result="NOT_ANALYZED",

        confidence=0.0,

        details={
            "message": (
                f"{media_type} analysis "
                "is not implemented yet."
            )
        }
    )