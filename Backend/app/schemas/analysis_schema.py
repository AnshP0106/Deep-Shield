from typing import Optional

from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    status: str
    media_type: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    explanation: Optional[str] = None

