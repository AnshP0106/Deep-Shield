from pathlib import Path
import numpy as np

from app.detectors.video_detector import video_detector
from app.managers.model_manager import model_manager


model_manager.load_models()

video_path = Path("test.mp4")

# 1. Extract Frames
frames = video_detector.extract_frames(video_path, sample_every_seconds=1.0)
print("\nTotal extracted frames:", len(frames))

# 2. Detect Faces & Run Tracking
detection_results = video_detector.detect_faces_in_frames(frames)

# 3. Analyze Faces (Xception)
analysis_results = video_detector.analyze_faces_in_frames(detection_results)

# =========================================================
# DETAILED XCEPTION DIAGNOSTIC AUDIT
# =========================================================
print("\n========================================")
print("FRAME-BY-FRAME XCEPTION PREDICTIONS")
print("========================================")

all_fake_probs = []
real_count = 0
fake_count = 0
uncertain_count = 0

for frame_res in analysis_results:
    f_num = frame_res["frame_number"]
    ts = frame_res["timestamp"]
    faces = frame_res.get("faces", [])
    
    print(f"\nFrame {f_num} (t={ts:.2f}s):")
    if not faces:
        print("    [No faces detected]")
        continue
        
    for face in faces:
        f_id = face["face_id"]
        r_prob = face["real_probability"]
        f_prob = face["fake_probability"]
        pred = face["prediction"]
        
        all_fake_probs.append(f_prob)
        if pred == "REAL":
            real_count += 1
        elif pred == "FAKE":
            fake_count += 1
        else:
            uncertain_count += 1
            
        print(f"    Face {f_id} -> {pred} (REAL: {r_prob * 100:.1f}%, FAKE: {f_prob * 100:.1f}%)")

print("\n----------------------------------------")
print("DISTRIBUTION SUMMARY")
print("----------------------------------------")
print(f"Total face predictions : {len(all_fake_probs)}")
print(f"REAL predictions       : {real_count}")
print(f"FAKE predictions       : {fake_count}")
print(f"UNCERTAIN predictions  : {uncertain_count}")

if all_fake_probs:
    print(f"Average fake prob      : {np.mean(all_fake_probs) * 100:.2f}%")
    print(f"Median fake prob       : {np.median(all_fake_probs) * 100:.2f}%")
    print(f"Maximum fake prob      : {np.max(all_fake_probs) * 100:.2f}%")
    print(f"Minimum fake prob      : {np.min(all_fake_probs) * 100:.2f}%")
else:
    print("No face predictions available for statistical analysis.")
print("----------------------------------------\n")

# 4. Video-Level Evidence Aggregation
video_evidence = video_detector.aggregate_video_evidence(analysis_results)

# 5. Temporal Evidence
temporal_evidence = video_detector.analyze_temporal_evidence(analysis_results)

# 6. Evidence Frame Extraction
suspicious_segments = temporal_evidence.get("suspicious_segments", [])
evidence_frames = video_detector.extract_evidence_frames(
    video_path=video_path,
    suspicious_segments=suspicious_segments,
    analyzed_frames=analysis_results
)

print("\n========================================")
print("FINAL CALIBRATED SUMMARY")
print("========================================")
print("Result:", video_evidence["visual_result"])
print("Risk level:", video_evidence["visual_risk_level"])
print(f"Risk score: {video_evidence['visual_risk_score'] * 100:.2f}% ({video_evidence['visual_risk_score']:.4f})")
print("Strong fake count:", video_evidence["strong_fake_count"])
print("Moderate fake count:", video_evidence["moderate_fake_count"])
print("Strong real count:", video_evidence["strong_real_count"])
print("Reason:", video_evidence["risk_reason"])
print("Evidence frames extracted:", len(evidence_frames))