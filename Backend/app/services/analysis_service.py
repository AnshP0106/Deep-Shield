from pathlib import Path
from typing import Union
import cv2

from app.schemas.analysis_schema import AnalysisResponse
from app.schemas.video_analysis_schema import (
    VideoAnalysisResponse,
    VideoMetadata,
    SuspiciousSegment,
    EvidenceFrame,
)

from app.detectors.image_detector import image_detector
from app.detectors.video_detector import video_detector
from app.detectors.audio_detector import analyze_audio


def analyze_media(
    file_path: Path,
    media_type: str
) -> Union[AnalysisResponse, VideoAnalysisResponse]:
    print("DEBUG media_type =", repr(media_type))

    # =====================================================
    # IMAGE ANALYSIS
    # =====================================================

    if media_type == "image":

        result = image_detector.predict_multiple_faces(
            file_path
        )

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
    # VIDEO ANALYSIS
    # =====================================================

    if media_type == "video":

        raw_result = video_detector.analyze_video(
            file_path
        )

        video_evidence = raw_result.get(
            "video_evidence",
            {}
        )

        temporal_evidence = raw_result.get(
            "temporal_evidence",
            {}
        )

        frame_level_analysis = raw_result.get(
            "frame_level_analysis",
            []
        )

        # -------------------------------------------------
        # Extract Video Technical Metadata
        # -------------------------------------------------

        cap = cv2.VideoCapture(
            str(file_path)
        )

        fps = (
            float(
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )
            if cap.isOpened()
            else 0.0
        )

        total_frames = (
            int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )
            if cap.isOpened()
            else 0
        )

        cap.release()

        if fps <= 0.0:
            fps = 30.0

        duration_seconds = (
            round(
                total_frames / fps,
                2
            )
            if fps > 0
            else 0.0
        )

        metadata_obj = VideoMetadata(
            total_frames=total_frames,
            analyzed_frames=len(
                frame_level_analysis
            ),
            fps=round(
                fps,
                2
            ),
            duration_seconds=duration_seconds
        )

        # -------------------------------------------------
        # Map Suspicious Segments
        # -------------------------------------------------

        suspicious_segments_list = []

        raw_segments = temporal_evidence.get(
            "suspicious_segments",
            []
        )

        for seg in raw_segments:

            suspicious_segments_list.append(
                SuspiciousSegment(
                    start_time=float(
                        seg.get(
                            "start_time",
                            0.0
                        )
                    ),

                    end_time=float(
                        seg.get(
                            "end_time",
                            0.0
                        )
                    ),

                    frame_count=int(
                        seg.get(
                            "frame_count",
                            0
                        )
                    ),

                    average_fake_probability=float(
                        seg.get(
                            "average_fake_probability",
                            0.0
                        )
                    ),

                    maximum_fake_probability=float(
                        seg.get(
                            "maximum_fake_probability",
                            0.0
                        )
                    ),

                    strength=str(
                        seg.get(
                            "strength",
                            "UNKNOWN"
                        )
                    )
                )
            )

        # -------------------------------------------------
        # Map Evidence Frames
        # -------------------------------------------------

        evidence_frames_list = []

        raw_evidence_frames = raw_result.get(
            "evidence_frames",
            []
        )

        for ev in raw_evidence_frames:

            evidence_frames_list.append(
                EvidenceFrame(
                    evidence_id=int(
                        ev.get(
                            "evidence_id",
                            0
                        )
                    ),

                    timestamp=float(
                        ev.get(
                            "timestamp",
                            0.0
                        )
                    ),

                    frame_index=int(
                        ev.get(
                            "frame_index",
                            0
                        )
                    ),

                    file_path=str(
                        ev.get(
                            "file_path",
                            ""
                        )
                    ),

                    annotated_file_path=ev.get(
                        "annotated_file_path"
                    ),

                    face_id=ev.get(
                        "face_id"
                    ),

                    fake_probability=ev.get(
                        "fake_probability"
                    ),

                    average_fake_probability=float(
                        ev.get(
                            "average_fake_probability",
                            0.0
                        )
                    ),

                    maximum_fake_probability=float(
                        ev.get(
                            "maximum_fake_probability",
                            0.0
                        )
                    ),

                    strength=str(
                        ev.get(
                            "strength",
                            "UNKNOWN"
                        )
                    )
                )
            )

        # -------------------------------------------------
        # Map Overall Metrics
        # -------------------------------------------------

        overall_fake_prob = round(
            float(
                video_evidence.get(
                    "visual_risk_score",
                    0.0
                )
            ) / 100.0,
            4
        )

        overall_real_prob = round(
            max(
                0.0,
                1.0 - overall_fake_prob
            ),
            4
        )

        overall_prediction = str(
            video_evidence.get(
                "visual_result",
                "UNCERTAIN"
            )
        )

        risk_level = str(
            video_evidence.get(
                "visual_risk_level",
                "UNKNOWN"
            )
        )

        # -------------------------------------------------
        # Unique Faces
        # -------------------------------------------------

        unique_face_ids = set()

        for f_res in frame_level_analysis:

            for face in f_res.get(
                "faces",
                []
            ):

                face_id = face.get(
                    "face_id"
                )

                if face_id is not None:
                    unique_face_ids.add(
                        face_id
                    )

        return VideoAnalysisResponse(
            media_type="VIDEO",

            overall_prediction=overall_prediction,

            overall_fake_probability=overall_fake_prob,

            overall_real_probability=overall_real_prob,

            risk_level=risk_level,

            video_metadata=metadata_obj,

            unique_faces_detected=len(
                unique_face_ids
            ),

            suspicious_segments=suspicious_segments_list,

            evidence_frames=evidence_frames_list
        )

    # =====================================================
    # AUDIO ANALYSIS
    # =====================================================

    if media_type == "audio":

        result = analyze_audio(
            file_path
        )

        # -------------------------------------------------
        # Handle detector failure
        # -------------------------------------------------

        if result.get(
            "detector_status"
        ) != "success":

            return AnalysisResponse(
                media_type=media_type,

                result="ANALYSIS_FAILED",

                confidence=0.0,

                details={
                    "message": (
                        "Audio analysis failed."
                    ),

                    "error": result.get(
                        "error",
                        "Unknown error"
                    )
                }
            )

        # -------------------------------------------------
        # Audio result
        # -------------------------------------------------

        classification = result.get(
            "classification"
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        ai_probability = float(
            result.get(
                "ai_probability",
                0.0
            )
        )

        human_probability = float(
            result.get(
                "human_probability",
                0.0
            )
        )

        # -------------------------------------------------
        # Return standardized AnalysisResponse
        # -------------------------------------------------

        return AnalysisResponse(
            media_type=media_type,

            result=classification,

            confidence=confidence,

            details={

                "ai_probability":
                    ai_probability,

                "human_probability":
                    human_probability,

                "duration_seconds":
                    result.get(
                        "audio",
                        {}
                    ).get(
                        "duration_seconds"
                    ),

                "sample_rate":
                    result.get(
                        "audio",
                        {}
                    ).get(
                        "sample_rate"
                    ),

                "num_samples":
                    result.get(
                        "audio",
                        {}
                    ).get(
                        "num_samples"
                    ),

                "rms":
                    result.get(
                        "audio",
                        {}
                    ).get(
                        "rms"
                    ),

                "zero_crossing_rate":
                    result.get(
                        "audio",
                        {}
                    ).get(
                        "zero_crossing_rate"
                    ),

                "detector":
                    "Deepfake Audio Detector"
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