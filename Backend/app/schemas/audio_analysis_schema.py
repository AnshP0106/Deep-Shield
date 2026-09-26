from typing import Optional

from pydantic import BaseModel


class AudioAnalysisResponse(BaseModel):
    file: str

    duration_seconds: float

    sample_rate: int

    num_samples: int

    model: str

    classification: Optional[str] = None

    confidence: Optional[float] = None

    spoof_probability: Optional[float] = None

    bonafide_probability: Optional[float] = None