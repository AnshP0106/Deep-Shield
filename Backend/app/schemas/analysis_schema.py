from typing import Any, Dict

from pydantic import BaseModel


class AnalysisResponse(BaseModel):

    media_type: str

    result: str

    confidence: float

    details: Dict[str, Any]