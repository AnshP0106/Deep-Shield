from pathlib import Path

from app.managers.model_manager import model_manager
from app.ml.preprocessing import preprocess_image
from app.schemas.analysis_schema import AnalysisResponse
from app.utils.image_utils import load_image


def analyze_image(file_path: Path) -> AnalysisResponse:

    model = model_manager.get_image_model()

    image = load_image(file_path)

    input_tensor = preprocess_image(image)

    width, height = image.size

    return AnalysisResponse(
        status="Preprocessing Complete",
        media_type="image",
        prediction=None,
        confidence=None,
        model_name="Xception",
        explanation=(
            f"Image preprocessing completed. "
            f"Original size: {width}x{height}."
        )
    )