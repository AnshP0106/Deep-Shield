from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from app.ml.preprocessing import preprocess_image
from app.managers.model_manager import model_manager


class ImageDetector:

    def __init__(self):
        self.device = model_manager.device

        # Load OpenCV face detectors
        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

        self.face_detector = cv2.CascadeClassifier(frontal_path)
        self.profile_face_detector = cv2.CascadeClassifier(profile_path)

        if self.face_detector.empty():
            raise RuntimeError(f"Failed to load frontal cascade: {frontal_path}")

        if self.profile_face_detector.empty():
            raise RuntimeError(f"Failed to load profile cascade: {profile_path}")

        # Initialize CLAHE for low-contrast/dark profile detection
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image around center to detect angled/tilted profiles."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)

    def _unrotate_bbox(
        self, bbox: tuple, angle: float, img_width: int, img_height: int
    ) -> tuple:
        """Converts bounding box coordinates back from rotated space to original image space."""
        x, y, w, h = bbox
        center_x, center_y = img_width / 2.0, img_height / 2.0

        # Calculate coordinates of box corners in rotated space
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

        # Reverse matrix rotation
        inv_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle, 1.0)
        corners_homo = np.hstack([corners, np.ones((4, 1))])
        original_corners = np.dot(inv_matrix, corners_homo.T).T

        min_x = max(0, int(np.min(original_corners[:, 0])))
        min_y = max(0, int(np.min(original_corners[:, 1])))
        max_x = min(img_width, int(np.max(original_corners[:, 0])))
        max_y = min(img_height, int(np.max(original_corners[:, 1])))

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def detect_face(self, file_path: Path):
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError("Unable to read image.")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Boost contrast for shadow-heavy profiles
            enhanced_gray = self.clahe.apply(gray)

            # 1. ALWAYS initialize faces as a mutable Python list
            faces = []
            detection_type = "none"

            # --------------------------------------------------
            # 1. Frontal Detection (Original + Enhanced)
            # --------------------------------------------------
            frontal_detected = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
            )
            if len(frontal_detected) > 0:
                faces = list(frontal_detected)
                detection_type = "frontal"

            if len(faces) == 0:
                frontal_detected_enh = self.face_detector.detectMultiScale(
                    enhanced_gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
                )
                if len(frontal_detected_enh) > 0:
                    faces = list(frontal_detected_enh)
                    detection_type = "frontal_enhanced"

            # --------------------------------------------------
            # 2. Profile Detection across Rotations & Flips
            # --------------------------------------------------
            if len(faces) == 0:
                rotation_angles = [0, -20, 20]
                
                for angle in rotation_angles:
                    rotated_gray = self._rotate_image(enhanced_gray, angle)
                    
                    for is_flipped in [False, True]:
                        target = cv2.flip(rotated_gray, 1) if is_flipped else rotated_gray
                        
                        detected = self.profile_face_detector.detectMultiScale(
                            target, scaleFactor=1.03, minNeighbors=3, minSize=(35, 35)
                        )

                        if len(detected) > 0:
                            img_h, img_w = target.shape[:2]
                            for bx, by, bw, bh in detected:
                                if is_flipped:
                                    bx = img_w - bx - bw
                                
                                orig_bbox = self._unrotate_bbox(
                                    (bx, by, bw, bh), angle, img_w, img_h
                                )
                                # Now faces is guaranteed to be a list
                                faces.append(orig_bbox)

                            detection_type = f"profile_rot_{angle}_flip_{is_flipped}"
                            break
                    
                    if len(faces) > 0:
                        break

            print("Detection type:", detection_type)
            print("Faces detected:", len(faces))

            if len(faces) == 0:
                return None

            # Select largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, width, height = largest_face

            # Dynamic Margin (20% expansion around profile bounds)
            margin_x = int(width * 0.20)
            margin_y = int(height * 0.20)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + width + margin_x)
            y2 = min(image.shape[0], y + height + margin_y)

            face_crop = image[y1:y2, x1:x2]
            cv2.imwrite(
            "debug_face_crop.jpg",
            face_crop
             )

            print("Face crop shape:", face_crop.shape)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            return Image.fromarray(face_crop)

    def predict(self, file_path: Path):
        face = self.detect_face(file_path)

        if face is None:
            return {
                "prediction": "NO_FACE",
                "confidence": 0.0,
                "real_probability": 0.0,
                "fake_probability": 0.0,
                "face_detected": False
            }

        image_tensor = preprocess_image(face).to(self.device)
        model = model_manager.get_image_model()

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)

        real_probability = probabilities[0][0].item()
        fake_probability = probabilities[0][1].item()

        if fake_probability > real_probability:
            prediction = "FAKE"
            confidence = fake_probability
        else:
            prediction = "REAL"
            confidence = real_probability

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "real_probability": round(real_probability * 100, 2),
            "fake_probability": round(fake_probability * 100, 2),
            "face_detected": True
        }


image_detector = ImageDetector()