import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import torch
from PIL import Image

from app.detectors.scrfd_detector import scrfd_detector
from app.managers.model_manager import model_manager
from app.ml.face_tracker import IoUFaceTracker
from app.ml.preprocessing import preprocess_image


class VideoDetector:

    def __init__(self):

        print(
            "Video detector initialized (Backend: SCRFD)."
        )

    # =========================================================
    # EXTRACT FRAMES
    # =========================================================

    def extract_frames(
        self,
        file_path: Path,
        sample_every_seconds: float = 1.0
    ):

        # =====================================================
        # OPEN VIDEO
        # =====================================================

        video = cv2.VideoCapture(
            str(file_path)
        )

        if not video.isOpened():

            raise ValueError(
                "Unable to open video."
            )

        # =====================================================
        # VIDEO INFORMATION
        # =====================================================

        fps = video.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = int(
            video.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            video.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            video.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        duration = (
            frame_count / fps
            if fps > 0
            else 0
        )

        print(
            "========================================"
        )

        print(
            "VIDEO INFORMATION"
        )

        print(
            "========================================"
        )

        print(
            "FPS:",
            fps
        )

        print(
            "Total frames:",
            frame_count
        )

        print(
            "Resolution:",
            f"{width}x{height}"
        )

        print(
            "Duration:",
            round(
                duration,
                2
            ),
            "seconds"
        )

        # =====================================================
        # DETERMINE SAMPLING INTERVAL
        # =====================================================

        if fps <= 0:

            video.release()

            raise ValueError(
                "Could not determine video FPS."
            )

        frame_interval = max(
            1,
            int(
                fps *
                sample_every_seconds
            )
        )

        print(
            "Sampling every",
            sample_every_seconds,
            "second(s)"
        )

        print(
            "Frame interval:",
            frame_interval
        )

        # =====================================================
        # FRAME EXTRACTION
        # =====================================================

        frames = []

        current_frame = 0

        while True:

            success, frame = (
                video.read()
            )

            if not success:

                break

            # -------------------------------------------------
            # Select sampled frames
            # -------------------------------------------------

            if (
                current_frame
                % frame_interval
                == 0
            ):

                timestamp = (
                    current_frame / fps
                )

                frames.append({

                    "frame_number":
                        current_frame,

                    "timestamp":
                        round(
                            timestamp,
                            3
                        ),

                    "image":
                        frame
                })

                print(
                    f"Extracted frame "
                    f"{current_frame} "
                    f"at "
                    f"{timestamp:.2f}s"
                )

            current_frame += 1

        video.release()

        print(
            "========================================"
        )

        print(
            "FRAME EXTRACTION COMPLETE"
        )

        print(
            "Frames extracted:",
            len(frames)
        )

        print(
            "========================================"
        )

        return frames

    # =========================================================
    # SCRFD FACE DETECTION & TRACKING
    # =========================================================

    def detect_faces_in_frames(
        self,
        frames
    ):
        tracker = IoUFaceTracker(
            iou_threshold=0.35,
            max_lost_frames=5
        )

        results = []

        print(
            "========================================"
        )

        print(
            "SCRFD VIDEO FACE DETECTION & TRACKING"
        )

        print(
            "Detector Backend: SCRFD (ONNX Runtime)"
        )

        print(
            "========================================"
        )

        for index, frame_data in enumerate(
            frames
        ):

            frame = frame_data[
                "image"
            ]

            frame_number = frame_data[
                "frame_number"
            ]

            timestamp = frame_data[
                "timestamp"
            ]

            print(
                f"\nProcessing frame "
                f"{index + 1}/{len(frames)}"
            )

            print(
                f"Frame number: "
                f"{frame_number}"
            )

            print(
                f"Timestamp: "
                f"{timestamp}s"
            )

            # ---------------------------------------------
            # Run SCRFD Face Detection
            # ---------------------------------------------

            raw_detections = scrfd_detector.detect_faces(
                frame
            )

            print(
                "Faces detected by SCRFD:",
                len(raw_detections)
            )

            # ---------------------------------------------
            # Filter detections before tracking
            # ---------------------------------------------

            filtered_detections: List[Dict[str, Any]] = []

            for face_index, det in enumerate(
                raw_detections
            ):

                x, y, width, height = det["bbox"]
                detection_confidence = float(det["confidence"])

                print(
                    f"Candidate {face_index + 1}: "
                    f"x={x}, "
                    f"y={y}, "
                    f"w={width}, "
                    f"h={height}, "
                    f"confidence="
                    f"{detection_confidence:.3f}"
                )

                # =========================================
                # FACE QUALITY FILTER #1
                # Minimum face size
                # =========================================

                if (
                    width < 40
                    or
                    height < 40
                ):

                    print(
                        f"Candidate {face_index + 1}: "
                        f"SKIPPED — face too small "
                        f"({width}x{height})"
                    )

                    continue

                # =========================================
                # FACE QUALITY FILTER #2
                # Minimum confidence
                # =========================================

                if detection_confidence < 0.50:

                    print(
                        f"Candidate {face_index + 1}: "
                        f"SKIPPED — detection confidence "
                        f"too low "
                        f"({detection_confidence:.3f})"
                    )

                    continue

                # =========================================
                # ACCEPT DETECTION
                # =========================================

                print(
                    f"Candidate {face_index + 1}: "
                    "ACCEPTED for tracking."
                )

                filtered_detections.append({
                    "bbox": [x, y, width, height],
                    "confidence": detection_confidence
                })

            # ---------------------------------------------
            # Update IoU Tracker
            # ---------------------------------------------

            tracked_faces = tracker.update(
                detections=filtered_detections,
                frame_idx=index,
                timestamp=timestamp
            )

            # ---------------------------------------------
            # Store tracked face information
            # ---------------------------------------------

            face_results = []

            for tracked in tracked_faces:

                face_id = tracked["face_id"]
                bx, by, bw, bh = tracked["bbox"]
                det_confidence = tracked["confidence"]

                print(
                    f"  Tracked Face ID {face_id}: "
                    f"x={bx}, y={by}, w={bw}, h={bh}, "
                    f"confidence={det_confidence:.3f}"
                )

                face_results.append({

                    "face_id":
                        face_id,

                    "x":
                        bx,

                    "y":
                        by,

                    "width":
                        bw,

                    "height":
                        bh,

                    "detection_confidence":
                        det_confidence
                })

            # ---------------------------------------------
            # Store frame result
            # ---------------------------------------------

            results.append({

                "frame_number":
                    frame_number,

                "timestamp":
                    timestamp,

                "image":
                    frame,

                "face_count":
                    len(face_results),

                "faces":
                    face_results
            })

        print(
            "\n========================================"
        )

        print(
            "SCRFD VIDEO FACE DETECTION & TRACKING COMPLETE"
        )

        print(
            "========================================"
        )

        return results

    # =========================================================
    # XCEPTION FACE ANALYSIS
    # =========================================================

    def analyze_faces_in_frames(
        self,
        frame_results
    ):

        device = model_manager.device

        model = model_manager.get_image_model()

        if model is None:

            raise RuntimeError(
                "Deepfake image model is not loaded."
            )

        model.eval()

        results = []

        print(
            "========================================"
        )

        print(
            "XCEPTION VIDEO FACE ANALYSIS"
        )

        print(
            "========================================"
        )

        for frame_result in frame_results:

            frame_number = frame_result[
                "frame_number"
            ]

            timestamp = frame_result[
                "timestamp"
            ]

            frame = frame_result[
                "image"
            ]

            faces = frame_result[
                "faces"
            ]

            print(
                f"\nAnalyzing frame "
                f"{frame_number} "
                f"at {timestamp}s"
            )

            analyzed_faces = []

            for face in faces:

                face_id = face["face_id"]

                x = face["x"]
                y = face["y"]

                width = face["width"]
                height = face["height"]

                detection_confidence = (
                    face["detection_confidence"]
                )

                # ---------------------------------------------
                # Clamped Bounding Box with Margin
                # ---------------------------------------------

                margin_x = int(
                    width * 0.20
                )

                margin_y = int(
                    height * 0.20
                )

                x1 = max(
                    0,
                    x - margin_x
                )

                y1 = max(
                    0,
                    y - margin_y
                )

                x2 = min(
                    frame.shape[1],
                    x + width + margin_x
                )

                y2 = min(
                    frame.shape[0],
                    y + height + margin_y
                )

                # Ensure non-degenerate dimensions
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:

                    print(
                        f"Face ID {face_id}: "
                        "Invalid crop dimensions."
                    )

                    continue

                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if face_crop.size == 0:

                    print(
                        f"Face ID {face_id}: "
                        "Invalid crop."
                    )

                    continue

                # ---------------------------------------------
                # BGR -> RGB
                # ---------------------------------------------

                face_crop_rgb = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )

                face_pil = Image.fromarray(
                    face_crop_rgb
                )

                # ---------------------------------------------
                # Preprocess
                # ---------------------------------------------

                image_tensor = (
                    preprocess_image(
                        face_pil
                    )
                    .to(device)
                )

                # ---------------------------------------------
                # Xception inference
                # ---------------------------------------------

                with torch.no_grad():

                    output = model(
                        image_tensor
                    )

                    probabilities = (
                        torch.softmax(
                            output,
                            dim=1
                        )
                    )

                real_probability = (
                    probabilities[0][0].item()
                )

                fake_probability = (
                    probabilities[0][1].item()
                )

                # ---------------------------------------------
                # Classification
                # ---------------------------------------------

                if fake_probability >= 0.75:

                    prediction = "FAKE"

                    confidence = (
                        fake_probability
                    )

                elif real_probability >= 0.75:

                    prediction = "REAL"

                    confidence = (
                        real_probability
                    )

                else:

                    prediction = "UNCERTAIN"

                    confidence = max(
                        real_probability,
                        fake_probability
                    )

                print(
                    f"Face ID {face_id}: "
                    f"{prediction} "
                    f"({confidence * 100:.2f}%)"
                )

                analyzed_faces.append({

                    "face_id":
                        face_id,

                    "x":
                        x,

                    "y":
                        y,

                    "width":
                        width,

                    "height":
                        height,

                    "detection_confidence":
                        round(
                            detection_confidence * 100,
                            2
                        ),

                    "prediction":
                        prediction,

                    "confidence":
                        round(
                            confidence * 100,
                            2
                        ),

                    "real_probability":
                        round(
                            real_probability * 100,
                            2
                        ),

                    "fake_probability":
                        round(
                            fake_probability * 100,
                            2
                        )
                })

            results.append({

                "frame_number":
                    frame_number,

                "timestamp":
                    timestamp,

                "image":
                    frame,

                "face_count":
                    len(analyzed_faces),

                "faces":
                    analyzed_faces
            })

        print(
            "\n========================================"
        )

        print(
            "XCEPTION VIDEO ANALYSIS COMPLETE"
        )

        print(
            "========================================"
        )

        return results

    # =========================================================
    # VIDEO-LEVEL EVIDENCE AGGREGATION
    # =========================================================

    def aggregate_video_evidence(
        self,
        analyzed_frames
    ):
        """
        Combines individual face predictions across video frames
        into one overall visual verdict.
        """

        print(
            "========================================"
        )

        print(
            "VIDEO-LEVEL EVIDENCE AGGREGATION"
        )

        print(
            "========================================"
        )

        strong_fake_faces = []
        moderate_fake_faces = []

        strong_real_faces = []
        moderate_real_faces = []

        uncertain_faces = []

        suspicious_timestamps = []

        total_faces_analyzed = 0

        for frame_result in analyzed_frames:

            timestamp = frame_result[
                "timestamp"
            ]

            faces = frame_result[
                "faces"
            ]

            for face in faces:

                total_faces_analyzed += 1

                prediction = face[
                    "prediction"
                ]

                fake_probability = (
                    face["fake_probability"]
                    / 100.0
                )

                real_probability = (
                    face["real_probability"]
                    / 100.0
                )

                if (
                    prediction == "FAKE"
                    and
                    fake_probability >= 0.90
                ):

                    strong_fake_faces.append(
                        face
                    )

                    suspicious_timestamps.append(
                        timestamp
                    )

                elif (
                    prediction == "FAKE"
                    and
                    fake_probability >= 0.75
                ):

                    moderate_fake_faces.append(
                        face
                    )

                    suspicious_timestamps.append(
                        timestamp
                    )

                elif (
                    prediction == "REAL"
                    and
                    real_probability >= 0.90
                ):

                    strong_real_faces.append(
                        face
                    )

                elif (
                    prediction == "REAL"
                    and
                    real_probability >= 0.75
                ):

                    moderate_real_faces.append(
                        face
                    )

                else:

                    uncertain_faces.append(
                        face
                    )

        suspicious_timestamps = sorted(
            list(
                set(
                    suspicious_timestamps
                )
            )
        )

        strong_fake_count = len(
            strong_fake_faces
        )

        moderate_fake_count = len(
            moderate_fake_faces
        )

        strong_real_count = len(
            strong_real_faces
        )

        moderate_real_count = len(
            moderate_real_faces
        )

        uncertain_count = len(
            uncertain_faces
        )

        print(
            "Total faces analyzed:",
            total_faces_analyzed
        )

        print(
            "Strong fake evidence:",
            strong_fake_count
        )

        print(
            "Moderate fake evidence:",
            moderate_fake_count
        )

        print(
            "Strong real evidence:",
            strong_real_count
        )

        print(
            "Moderate real evidence:",
            moderate_real_count
        )

        print(
            "Uncertain evidence:",
            uncertain_count
        )

        print(
            "Suspicious timestamps:",
            suspicious_timestamps
        )

        fake_score = (
            strong_fake_count * 3.0
            +
            moderate_fake_count * 1.5
        )

        real_score = (
            strong_real_count * 1.0
            +
            moderate_real_count * 0.5
        )

        total_evidence = (
            fake_score
            +
            real_score
        )

        if total_evidence > 0:

            visual_risk_score = (
                fake_score
                /
                total_evidence
            ) * 100

        else:

            visual_risk_score = 0.0

        if strong_fake_count >= 1:

            visual_result = (
                "LIKELY_FAKE"
            )

            visual_risk_level = (
                "HIGH"
            )

            risk_reason = (
                "Strong facial manipulation "
                "evidence was detected in "
                "one or more video frames."
            )

        elif moderate_fake_count >= 2:

            visual_result = (
                "LIKELY_FAKE"
            )

            visual_risk_level = (
                "HIGH"
            )

            risk_reason = (
                "Multiple video frames "
                "contained evidence "
                "consistent with facial "
                "manipulation."
            )

        elif (
            moderate_fake_count == 1
            or
            uncertain_count > 0
        ):

            visual_result = (
                "UNCERTAIN"
            )

            visual_risk_level = (
                "MEDIUM"
            )

            risk_reason = (
                "Some frames contained "
                "uncertain or suspicious "
                "facial predictions, but "
                "there is not enough strong "
                "evidence for a definitive "
                "fake classification."
            )

        else:

            visual_result = (
                "LIKELY_REAL"
            )

            visual_risk_level = (
                "LOW"
            )

            risk_reason = (
                "No strong evidence of "
                "facial manipulation was "
                "detected."
            )

        result = {

            "visual_result":
                visual_result,

            "visual_risk_level":
                visual_risk_level,

            "visual_risk_score":
                round(
                    visual_risk_score,
                    2
                ),

            "total_faces_analyzed":
                total_faces_analyzed,

            "strong_fake_count":
                strong_fake_count,

            "moderate_fake_count":
                moderate_fake_count,

            "strong_real_count":
                strong_real_count,

            "moderate_real_count":
                moderate_real_count,

            "uncertain_count":
                uncertain_count,

            "suspicious_timestamps":
                suspicious_timestamps,

            "risk_reason":
                risk_reason
        }

        print(
            "========================================"
        )

        print(
            "VIDEO EVIDENCE RESULT"
        )

        print(
            "========================================"
        )

        print(
            "Visual result:",
            visual_result
        )

        print(
            "Risk level:",
            visual_risk_level
        )

        print(
            "Risk score:",
            f"{visual_risk_score:.2f}"
        )

        print(
            "Reason:",
            risk_reason
        )

        print(
            "========================================"
        )

        return result

    # =========================================================
    # TEMPORAL EVIDENCE ANALYSIS
    # =========================================================

    def analyze_temporal_evidence(
        self,
        analyzed_frames
    ):
        """
        Detects consecutive suspicious predictions across
        the video timeline.
        """

        print(
            "========================================"
        )

        print(
            "TEMPORAL EVIDENCE ANALYSIS"
        )

        print(
            "========================================"
        )

        timeline = []

        for frame_result in analyzed_frames:

            timestamp = frame_result[
                "timestamp"
            ]

            faces = frame_result[
                "faces"
            ]

            if not faces:
                continue

            strongest_fake = max(
                faces,
                key=lambda face:
                    face["fake_probability"]
            )

            fake_probability = (
                strongest_fake[
                    "fake_probability"
                ]
            )

            if fake_probability >= 90:

                status = "STRONG_FAKE"

            elif fake_probability >= 75:

                status = "MODERATE_FAKE"

            else:

                status = "NOT_FAKE"

            timeline.append({

                "timestamp":
                    timestamp,

                "status":
                    status,

                "fake_probability":
                    fake_probability
            })

        suspicious_segments = []

        current_segment = []

        for item in timeline:

            if item["status"] in (
                "STRONG_FAKE",
                "MODERATE_FAKE"
            ):

                if not current_segment:

                    current_segment = [
                        item
                    ]

                else:

                    previous_timestamp = (
                        current_segment[-1][
                            "timestamp"
                        ]
                    )

                    current_timestamp = (
                        item[
                            "timestamp"
                        ]
                    )

                    if (
                        current_timestamp
                        -
                        previous_timestamp
                        <= 1.5
                    ):

                        current_segment.append(
                            item
                        )

                    else:

                        suspicious_segments.append(
                            current_segment
                        )

                        current_segment = [
                            item
                        ]

            else:

                if current_segment:

                    suspicious_segments.append(
                        current_segment
                    )

                    current_segment = []

        if current_segment:

            suspicious_segments.append(
                current_segment
            )

        temporal_segments = []

        for segment in suspicious_segments:

            start_time = segment[0][
                "timestamp"
            ]

            end_time = segment[-1][
                "timestamp"
            ]

            frame_count = len(
                segment
            )

            average_fake_probability = (
                sum(
                    item[
                        "fake_probability"
                    ]
                    for item in segment
                )
                /
                frame_count
            )

            maximum_fake_probability = max(
                item[
                    "fake_probability"
                ]
                for item in segment
            )

            if (
                frame_count >= 3
                and
                average_fake_probability >= 85
            ):

                strength = (
                    "EXTREMELY_STRONG"
                )

            elif (
                frame_count >= 2
                and
                average_fake_probability >= 85
            ):

                strength = (
                    "VERY_STRONG"
                )

            elif (
                average_fake_probability >= 90
            ):

                strength = (
                    "STRONG"
                )

            else:

                strength = (
                    "MODERATE"
                )

            temporal_segments.append({

                "start_time":
                    start_time,

                "end_time":
                    end_time,

                "frame_count":
                    frame_count,

                "average_fake_probability":
                    round(
                        average_fake_probability,
                        2
                    ),

                "maximum_fake_probability":
                    round(
                        maximum_fake_probability,
                        2
                    ),

                "strength":
                    strength
            })

        temporal_score = 0.0

        for segment in temporal_segments:

            frame_count = segment[
                "frame_count"
            ]

            average_probability = (
                segment[
                    "average_fake_probability"
                ]
            )

            segment_score = (
                average_probability
                / 100
            )

            if frame_count >= 3:

                multiplier = 1.50

            elif frame_count == 2:

                multiplier = 1.25

            else:

                multiplier = 1.00

            segment_score *= multiplier

            segment_score = min(
                segment_score,
                1.0
            )

            temporal_score = max(
                temporal_score,
                segment_score
            )

        temporal_score *= 100

        if temporal_score >= 85:

            temporal_result = (
                "STRONG_MANIPULATION_EVIDENCE"
            )

        elif temporal_score >= 70:

            temporal_result = (
                "MODERATE_MANIPULATION_EVIDENCE"
            )

        elif temporal_score >= 50:

            temporal_result = (
                "WEAK_MANIPULATION_EVIDENCE"
            )

        else:

            temporal_result = (
                "NO_STRONG_TEMPORAL_EVIDENCE"
            )

        print(
            "Temporal suspicious segments:",
            len(
                temporal_segments
            )
        )

        for segment in temporal_segments:

            print(
                f"Segment "
                f"{segment['start_time']}s — "
                f"{segment['end_time']}s | "
                f"Frames: "
                f"{segment['frame_count']} | "
                f"Average fake: "
                f"{segment['average_fake_probability']}% | "
                f"Strength: "
                f"{segment['strength']}"
            )

        print(
            "Temporal score:",
            round(
                temporal_score,
                2
            )
        )

        print(
            "Temporal result:",
            temporal_result
        )

        print(
            "========================================"
        )

        return {

            "temporal_score":
                round(
                    temporal_score,
                    2
                ),

            "temporal_result":
                temporal_result,

            "suspicious_segments":
                temporal_segments
        }

    # =========================================================
    # EVIDENCE FRAME ANNOTATION
    # =========================================================

    def annotate_evidence_frame(
        self,
        frame,
        face_bbox,
        timestamp,
        face_id,
        fake_probability,
        strength
    ):
        """
        Draws a high-contrast bounding box around the suspicious face and
        overlays analysis metadata at the bottom with a semi-transparent HUD banner.
        """
        annotated = frame.copy()
        img_h, img_w = annotated.shape[:2]

        # Draw Face Bounding Box if valid
        if face_bbox and len(face_bbox) == 4:
            bx, by, bw, bh = face_bbox
            x1 = max(0, min(int(bx), img_w - 1))
            y1 = max(0, min(int(by), img_h - 1))
            x2 = max(0, min(int(bx + bw), img_w - 1))
            y2 = max(0, min(int(by + bh), img_h - 1))

            if (x2 - x1) > 0 and (y2 - y1) > 0:
                # Primary Red Box (BGR: 0, 0, 240) with black border outline for visibility
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 240), 2)

                label = f"Face ID: {face_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                font_thickness = 1
                (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                label_y1 = max(0, y1 - label_h - 8)
                label_y2 = y1

                # Label background tag
                cv2.rectangle(annotated, (x1, label_y1), (x1 + label_w + 10, label_y2), (0, 0, 240), -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, label_y2 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )

        # Bottom HUD Banner
        banner_h = max(110, int(img_h * 0.16))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, img_h - banner_h), (img_w, img_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

        # Highlight strip (Orange/Red)
        cv2.line(annotated, (0, img_h - banner_h), (img_w, img_h - banner_h), (0, 70, 230), 2)

        # Format Timestamp (mm:ss.s)
        total_seconds = float(timestamp)
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        formatted_time = f"{minutes:02d}:{seconds:04.1f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1

        # Text Lines
        lines = [
            ("DeepShield Visual Forensics", (0, 190, 255), 0.65, 2),
            (f"Timestamp: {formatted_time} ({total_seconds:.2f}s)   |   Face ID: {face_id}", (240, 240, 240), 0.50, 1),
            (f"Fake Probability: {fake_probability:.1f}%   |   Evidence: {strength}", (0, 90, 255), 0.52, 2)
        ]

        text_y = img_h - banner_h + 26
        for text, color, scale, thick in lines:
            # Shadow outline
            cv2.putText(annotated, text, (20, text_y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            # Fore text
            cv2.putText(annotated, text, (20, text_y), font, scale, color, thick, cv2.LINE_AA)
            text_y += 26

        return annotated

    # =========================================================
    # SUSPICIOUS FRAME EXTRACTION
    # =========================================================

    def extract_evidence_frames(
        self,
        video_path,
        suspicious_segments,
        output_dir="uploads/video_evidence",
        analyzed_frames=None
    ):
        """
        Extracts representative evidence frames from detected suspicious segments,
        finds the most suspicious face associated with the segment, and saves both
        raw and annotated JPEG evidence files.
        """

        print(
            "========================================"
        )

        print(
            "EXTRACTING EVIDENCE FRAMES"
        )

        print(
            "========================================"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        video = cv2.VideoCapture(
            str(video_path)
        )

        if not video.isOpened():
            print(
                f"Failed to open video for evidence extraction: {video_path}"
            )
            return []

        fps = video.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            video.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:
            fps = 30.0

        evidence_frames = []

        try:
            for seg_idx, segment in enumerate(suspicious_segments, start=1):
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                seg_strength = segment.get("strength", "UNKNOWN")

                # -------------------------------------------------------------
                # Identify the face with the strongest fake probability
                # in analyzed frames within or adjacent to this segment
                # -------------------------------------------------------------
                best_face = None
                best_face_time = None
                highest_fake_prob = -1.0

                if analyzed_frames:
                    for f_res in analyzed_frames:
                        f_time = f_res.get("timestamp", 0.0)
                        # Check frames falling within [start_time - 0.25, end_time + 0.25]
                        if (start_time - 0.25) <= f_time <= (end_time + 0.25):
                            for face in f_res.get("faces", []):
                                f_fake = face.get("fake_probability", 0.0)
                                if f_fake > highest_fake_prob:
                                    highest_fake_prob = f_fake
                                    best_face = face
                                    best_face_time = f_time

                # Determine representative timestamp
                if start_time != end_time:
                    representative_timestamp = round(
                        (start_time + end_time) / 2.0,
                        3
                    )
                else:
                    representative_timestamp = start_time

                if representative_timestamp < 0:
                    representative_timestamp = 0.0

                target_frame_index = int(
                    representative_timestamp * fps
                )

                if total_frames > 0 and target_frame_index >= total_frames:
                    target_frame_index = max(0, total_frames - 1)

                print(
                    f"\nExtracting evidence frame at {representative_timestamp}s"
                )

                video.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    target_frame_index
                )

                success, frame = video.read()

                # Fallback: Seek backwards slightly if reading near the end failed
                if (not success or frame is None) and target_frame_index > 0:
                    fallback_index = max(0, target_frame_index - 1)
                    video.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        fallback_index
                    )
                    success, frame = video.read()
                    if success and frame is not None:
                        target_frame_index = fallback_index

                if not success or frame is None:
                    print(
                        f"Failed to read frame at {representative_timestamp}s "
                        f"(frame index: {target_frame_index})"
                    )
                    continue

                # -------------------------------------------------------------
                # 1. Save Raw Evidence Image
                # -------------------------------------------------------------
                filename = f"evidence_{seg_idx:03d}.jpg"
                file_path = os.path.join(
                    output_dir,
                    filename
                )

                saved = cv2.imwrite(
                    file_path,
                    frame
                )

                if not saved:
                    print(
                        f"Failed to save evidence frame to {file_path}"
                    )
                    continue

                print(
                    "Saved evidence frame:"
                )
                print(
                    file_path
                )

                # -------------------------------------------------------------
                # 2. Annotate and Save Annotated Evidence Image
                # -------------------------------------------------------------
                if best_face:
                    face_id = best_face.get("face_id", 1)
                    face_bbox = [
                        best_face.get("x", 0),
                        best_face.get("y", 0),
                        best_face.get("width", 0),
                        best_face.get("height", 0)
                    ]
                    face_fake_prob = best_face.get("fake_probability", segment.get("maximum_fake_probability", 0.0))
                else:
                    face_id = 1
                    face_bbox = None
                    face_fake_prob = segment.get("maximum_fake_probability", 0.0)

                annotated_frame = self.annotate_evidence_frame(
                    frame=frame,
                    face_bbox=face_bbox,
                    timestamp=representative_timestamp,
                    face_id=face_id,
                    fake_probability=face_fake_prob,
                    strength=seg_strength
                )

                annotated_filename = f"evidence_{seg_idx:03d}_annotated.jpg"
                annotated_file_path = os.path.join(
                    output_dir,
                    annotated_filename
                )

                cv2.imwrite(
                    annotated_file_path,
                    annotated_frame
                )

                print(
                    "Saved annotated evidence frame:"
                )
                print(
                    annotated_file_path
                )

                evidence_frames.append({
                    "evidence_id": seg_idx,
                    "timestamp": representative_timestamp,
                    "frame_index": target_frame_index,
                    "file_path": file_path,
                    "annotated_file_path": annotated_file_path,
                    "face_id": face_id,
                    "fake_probability": face_fake_prob,
                    "average_fake_probability": segment["average_fake_probability"],
                    "maximum_fake_probability": segment["maximum_fake_probability"],
                    "strength": seg_strength
                })

        finally:
            video.release()

        print(
            f"\nEvidence frames extracted: {len(evidence_frames)}"
        )

        print(
            "========================================"
        )

        return evidence_frames

    # =========================================================
    # ANALYZE VIDEO (FULL PIPELINE)
    # =========================================================

    def analyze_video(
        self,
        video_path: Path,
        sample_every_seconds: float = 1.0,
        evidence_output_dir: str = "uploads/video_evidence"
    ) -> Dict[str, Any]:
        """
        Executes full end-to-end video analysis:
        1. Extract frames (1-second sampling)
        2. Detect & track faces via SCRFD + IoU Face Tracker
        3. Analyze face crops with Xception
        4. Aggregate video visual evidence
        5. Analyze temporal sequences
        6. Extract suspicious representative evidence frames (raw & annotated)
        """
        video_path = Path(video_path)

        frames = self.extract_frames(
            video_path,
            sample_every_seconds=sample_every_seconds
        )

        detection_results = self.detect_faces_in_frames(frames)

        analysis_results = self.analyze_faces_in_frames(detection_results)

        video_evidence = self.aggregate_video_evidence(analysis_results)

        temporal_evidence = self.analyze_temporal_evidence(analysis_results)

        evidence_frames = self.extract_evidence_frames(
            video_path=video_path,
            suspicious_segments=temporal_evidence.get("suspicious_segments", []),
            output_dir=evidence_output_dir,
            analyzed_frames=analysis_results
        )

        return {
            "media_type": "VIDEO",
            "detector_backend": "SCRFD",
            "video_path": str(video_path),
            "video_evidence": video_evidence,
            "temporal_evidence": temporal_evidence,
            "evidence_frames": evidence_frames,
            "frame_level_analysis": analysis_results
        }


video_detector = VideoDetector()