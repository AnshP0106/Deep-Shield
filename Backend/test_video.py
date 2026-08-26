from pathlib import Path

from app.detectors.video_detector import (
    video_detector
)

from app.managers.model_manager import (
    model_manager
)


model_manager.load_models()


video_path = Path(
    "test.mp4"
)


# =====================================================
# STEP 1 — EXTRACT FRAMES
# =====================================================

frames = video_detector.extract_frames(
    video_path,
    sample_every_seconds=1.0
)


print(
    "\nTotal extracted frames:",
    len(frames)
)


# =====================================================
# STEP 2 — DETECT FACES
# =====================================================

results = (
    video_detector.detect_faces_in_frames(
        frames
    )
)

analysis_results = (
    video_detector.analyze_faces_in_frames(
        results
    )
)

# =========================================================
# VIDEO-LEVEL EVIDENCE
# =========================================================

video_evidence = (
    video_detector.aggregate_video_evidence(
        analysis_results
    )
)

# =========================================================
# TEMPORAL EVIDENCE
# =========================================================

temporal_evidence = (
    video_detector.analyze_temporal_evidence(
        analysis_results
    )
)

print(
    "\n========================================"
)

print(
    "FINAL TEMPORAL EVIDENCE"
)

print(
    "========================================"
)

print(
    "Temporal result:",
    temporal_evidence[
        "temporal_result"
    ]
)

print(
    "Temporal score:",
    temporal_evidence[
        "temporal_score"
    ]
)

print(
    "Suspicious segments:",
    temporal_evidence[
        "suspicious_segments"
    ]
)


# =========================================================
# EVIDENCE FRAME EXTRACTION
# =========================================================

suspicious_segments = temporal_evidence.get(
    "suspicious_segments",
    []
)

evidence_frames = video_detector.extract_evidence_frames(
    video_path=video_path,
    suspicious_segments=suspicious_segments
)

print(
    "\n========================================"
)

print(
    "EXTRACTED EVIDENCE FRAMES"
)

print(
    "========================================"
)

if evidence_frames:
    for evidence in evidence_frames:
        print(
            f"Evidence #{evidence['evidence_id']} "
            f"| Time: {evidence['timestamp']}s "
            f"| Frame Index: {evidence['frame_index']} "
            f"| Strength: {evidence['strength']}"
        )
        print(
            f"Path: {evidence['file_path']}\n"
        )
else:
    print(
        "No evidence frames generated."
    )


print(
    "\n========================================"
)

print(
    "FINAL VIDEO EVIDENCE"
)

print(
    "========================================"
)

print(
    "Result:",
    video_evidence["visual_result"]
)

print(
    "Risk level:",
    video_evidence["visual_risk_level"]
)

print(
    "Risk score:",
    video_evidence["visual_risk_score"]
)

print(
    "Strong fake frames:",
    video_evidence["strong_fake_count"]
)

print(
    "Moderate fake frames:",
    video_evidence["moderate_fake_count"]
)

print(
    "Uncertain:",
    video_evidence["uncertain_count"]
)

print(
    "Suspicious timestamps:",
    video_evidence[
        "suspicious_timestamps"
    ]
)

print(
    "Reason:",
    video_evidence["risk_reason"]
)

print(
    "\n========================================"
)

print(
    "FINAL VIDEO ANALYSIS"
)

print(
    "========================================"
)

for result in analysis_results:

    print(
        f"\nFrame {result['frame_number']} "
        f"| {result['timestamp']}s"
    )

    print(
        "Faces:",
        result["face_count"]
    )

    for face in result["faces"]:

        print(
            f"  Face {face['face_id']}  "
            f"{face['prediction']} "
            f"({face['confidence']}%)"
        )