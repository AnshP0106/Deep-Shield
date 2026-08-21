from pathlib import Path

import cv2


class YuNetDetector:

    def __init__(self):

        model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "weights"
            / "face_detection_yunet_2023mar.onnx"
        )

        print("Loading YuNet face detector...")
        print("YuNet model path:", model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

        print("YuNet face detector loaded successfully!")

    def detect(self, image):

        height, width = image.shape[:2]

        # YuNet needs the image size
        self.detector.setInputSize(
            (width, height)
        )

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        return faces


yunet_detector = YuNetDetector()