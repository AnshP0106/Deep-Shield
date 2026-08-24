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
            f"  Face {face['face_id']} → "
            f"{face['prediction']} "
            f"({face['confidence']}%)"
        )
