from typing import List, Optional
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    total_frames: int = Field(..., description="Total frame count of the video")
    analyzed_frames: int = Field(..., description="Number of sampled frames analyzed")
    fps: float = Field(..., description="Frames per second")
    duration_seconds: float = Field(..., description="Duration of the video in seconds")


class SuspiciousSegment(BaseModel):
    start_time: float = Field(..., description="Start timestamp of the suspicious segment in seconds")
    end_time: float = Field(..., description="End timestamp of the suspicious segment in seconds")
    frame_count: int = Field(..., description="Number of frames contained in the suspicious window")
    average_fake_probability: float = Field(..., description="Mean fake probability across the segment")
    maximum_fake_probability: float = Field(..., description="Peak fake probability observed in the segment")
    strength: str = Field(..., description="Evidence strength classification (e.g., VERY_STRONG, MODERATE)")


class EvidenceFrame(BaseModel):
    evidence_id: int = Field(..., description="Unique sequential identifier for the evidence snapshot")
    timestamp: float = Field(..., description="Representative timestamp of the evidence frame in seconds")
    frame_index: int = Field(..., description="Exact frame index captured from the video")
    file_path: str = Field(..., description="Relative storage path to the raw evidence frame")
    annotated_file_path: Optional[str] = Field(None, description="Relative storage path to the annotated frame overlay")
    face_id: Optional[int] = Field(None, description="Persistent track ID of the suspicious face if localized")
    fake_probability: Optional[float] = Field(None, description="Fake probability of the primary localized face")
    average_fake_probability: float = Field(..., description="Mean fake probability of the parent segment")
    maximum_fake_probability: float = Field(..., description="Max fake probability of the parent segment")
    strength: str = Field(..., description="Evidence severity level for this extracted snapshot")


class VideoAnalysisResponse(BaseModel):
    media_type: str = Field(default="VIDEO", description="Type of processed media")
    overall_prediction: str = Field(..., description="Final visual classification (e.g., LIKELY_FAKE, LIKELY_REAL, UNCERTAIN)")
    overall_fake_probability: float = Field(..., description="Aggregated confidence percentage that the video is fake")
    overall_real_probability: float = Field(..., description="Aggregated confidence percentage that the video is real")
    risk_level: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    video_metadata: VideoMetadata = Field(..., description="Technical metadata of the analyzed video")
    unique_faces_detected: int = Field(..., description="Count of distinct persistent face tracks identified")
    suspicious_segments: List[SuspiciousSegment] = Field(default_factory=list, description="Timeline intervals showing anomalous manipulations")
    evidence_frames: List[EvidenceFrame] = Field(default_factory=list, description="Extracted representative evidence snapshots")