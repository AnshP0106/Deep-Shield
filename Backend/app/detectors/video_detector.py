from pathlib import Path

from app.detectors.yunet_detector import yunet_detector

import torch
from PIL import Image
import cv2

from app.ml.preprocessing import preprocess_image
from app.managers.model_manager import model_manager


class VideoDetector:

    def __init__(self):

        print(
            "Video detector initialized."
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
    # YUNET FACE DETECTION
    # =========================================================

    def detect_faces_in_frames(
        self,
        frames
    ):

        results = []

        print(
            "========================================"
        )

        print(
            "YU NET VIDEO FACE DETECTION"
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
            # Run YuNet
            # ---------------------------------------------

            faces = yunet_detector.detect(
                frame
            )

            print(
                "Faces detected:",
                len(faces)
            )

            # ---------------------------------------------
            # Store face information
            # ---------------------------------------------

            face_results = []

            for face_index, face in enumerate(
                faces
            ):

                x, y, width, height = (
                    face[:4]
                )

                detection_confidence = (
                    face[-1]
                )

                x = int(x)
                y = int(y)
                width = int(width)
                height = int(height)

                detection_confidence = float(
                    detection_confidence
                )

                print(
                    f"Face {face_index + 1}: "
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
                        f"Face {face_index + 1}: "
                        f"SKIPPED — face too small "
                        f"({width}x{height})"
                    )

                    continue

                # =========================================
                # FACE QUALITY FILTER #2
                # Minimum YuNet confidence
                # =========================================

                if detection_confidence < 0.70:

                    print(
                        f"Face {face_index + 1}: "
                        f"SKIPPED — YuNet confidence "
                        f"too low "
                        f"({detection_confidence:.3f})"
                    )

                    continue

                # =========================================
                # ACCEPT FACE
                # =========================================

                print(
                    f"Face {face_index + 1}: "
                    "ACCEPTED for Xception analysis."
                )

                face_results.append({

                    "face_id":
                        face_index + 1,

                    "x":
                        x,

                    "y":
                        y,

                    "width":
                        width,

                    "height":
                        height,

                    "detection_confidence":
                        detection_confidence
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
            "VIDEO FACE DETECTION COMPLETE"
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

            for face_index, face in enumerate(
                faces
            ):

                x = face["x"]
                y = face["y"]

                width = face["width"]
                height = face["height"]

                detection_confidence = (
                    face["detection_confidence"]
                )

                # ---------------------------------------------
                # Crop face
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

                face_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if face_crop.size == 0:

                    print(
                        f"Face {face_index + 1}: "
                        "Invalid crop."
                    )

                    continue

                # ---------------------------------------------
                # BGR → RGB
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
                    f"Face {face_index + 1}: "
                    f"{prediction} "
                    f"({confidence * 100:.2f}%)"
                )

                analyzed_faces.append({

                    "face_id":
                        face_index + 1,

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


video_detector = VideoDetector()