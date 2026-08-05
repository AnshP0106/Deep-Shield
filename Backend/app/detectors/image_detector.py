from pathlib import Path

from app.managers.model_manager import model_manager
from app.schemas.analysis_schema import AnalysisResponse
from app.utils.image_utils import load_image


def analyze_image(file_path: Path) -> AnalysisResponse:
    """
    Image Deepfake Detection Pipeline.
    """

    # Step 1
    model = model_manager.get_image_model()

    # Step 2
    image = load_image(file_path)

    # Step 3
    width, height = image.size

    return AnalysisResponse(
        status="Pending",
        media_type="image",
        prediction=None,
        confidence=None,
        model_name=str(model),
        explanation=f"Image loaded successfully ({width}x{height}). AI model integration pending."
    )