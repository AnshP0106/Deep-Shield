from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.ml.preprocessing import preprocess_image
from app.managers.model_manager import model_manager
from app.detectors.yunet_detector import yunet_detector
from app.ml.gradcam import GradCAM

from app.utils.risk_scoring import (
    calculate_risk_level,
    calculate_overall_risk
)

def calculate_overall_risk(face_results):

    if not face_results:
        return {
            "risk_score": 0.0,
            "risk_level": "VERY_LOW",
            "reason": "No faces were detected."
        }

    total_faces = len(face_results)

    # ---------------------------------------------------------
    # Strong fake faces
    # ---------------------------------------------------------

    strong_fake_faces = [
        face
        for face in face_results
        if (
            face["prediction"] == "FAKE"
            and face["fake_probability"] >= 75.0
        )
    ]

    # ---------------------------------------------------------
    # Uncertain faces
    # ---------------------------------------------------------

    uncertain_faces = [
        face
        for face in face_results
        if face["prediction"] == "UNCERTAIN"
    ]

    # ---------------------------------------------------------
    # Fake face ratio
    # ---------------------------------------------------------

    fake_ratio = (
        len(strong_fake_faces)
        / total_faces
    )

    # ---------------------------------------------------------
    # Average fake probability
    # ---------------------------------------------------------

    if strong_fake_faces:

        average_fake_probability = sum(
            face["fake_probability"]
            for face in strong_fake_faces
        ) / len(strong_fake_faces)

    else:

        average_fake_probability = 0.0

    # ---------------------------------------------------------
    # Base risk score
    # ---------------------------------------------------------

    risk_score = (
        fake_ratio * 60
        +
        (average_fake_probability / 100) * 40
    )

    # ---------------------------------------------------------
    # Uncertainty contribution
    # ---------------------------------------------------------

    uncertainty_ratio = (
        len(uncertain_faces)
        / total_faces
    )

    risk_score += (
        uncertainty_ratio * 15
    )

    # ---------------------------------------------------------
    # Clamp score between 0 and 100
    # ---------------------------------------------------------

    risk_score = min(
        100.0,
        max(
            0.0,
            risk_score
        )
    )

    # ---------------------------------------------------------
    # Risk level
    # ---------------------------------------------------------

    if risk_score >= 80:

        risk_level = "VERY_HIGH"

    elif risk_score >= 60:

        risk_level = "HIGH"

    elif risk_score >= 35:

        risk_level = "MODERATE"

    elif risk_score >= 15:

        risk_level = "LOW"

    else:

        risk_level = "VERY_LOW"

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    if len(strong_fake_faces) > 0:

        reason = (
            f"{len(strong_fake_faces)} face(s) produced "
            f"strong evidence of manipulation."
        )

    elif len(uncertain_faces) > 0:

        reason = (
            f"{len(uncertain_faces)} face(s) produced "
            f"uncertain model predictions."
        )

    else:

        reason = (
            "No strong evidence of facial manipulation "
            "was detected."
        )

    return {
        "risk_score": round(
            risk_score,
            2
        ),
        "risk_level": risk_level,
        "reason": reason
    }

class ImageDetector:

    FAKE_THRESHOLD = 0.75
    REAL_THRESHOLD = 0.75
    UNCERTAIN_THRESHOLD = 0.60

    def __init__(self):

        # =====================================================
        # DEVICE
        # =====================================================

        self.device = model_manager.device

        # Grad-CAM will be initialized lazily
        self.gradcam = None

        # =====================================================
        # HAAR FALLBACK DETECTORS
        # =====================================================

        self.face_detector = None
        self.profile_face_detector = None

        try:

            if hasattr(cv2, "CascadeClassifier"):

                frontal_path = (
                    cv2.data.haarcascades
                    + "haarcascade_frontalface_default.xml"
                )

                profile_path = (
                    cv2.data.haarcascades
                    + "haarcascade_profileface.xml"
                )

                self.face_detector = (
                    cv2.CascadeClassifier(
                        frontal_path
                    )
                )

                self.profile_face_detector = (
                    cv2.CascadeClassifier(
                        profile_path
                    )
                )

                if (
                    self.face_detector.empty()
                    or self.profile_face_detector.empty()
                ):

                    print(
                        "WARNING: Haar cascade files "
                        "could not be loaded."
                    )

                    self.face_detector = None
                    self.profile_face_detector = None

                else:

                    print(
                        "Haar fallback detectors "
                        "loaded successfully!"
                    )

            else:

                print(
                    "WARNING: CascadeClassifier is "
                    "not available in this OpenCV build."
                )

                print(
                    "Haar fallback disabled. "
                    "YuNet will be used as the "
                    "primary detector."
                )

        except Exception as e:

            print(
                "WARNING: Could not initialize "
                f"Haar fallback: {e}"
            )

            self.face_detector = None
            self.profile_face_detector = None

        # =====================================================
        # CLAHE
        # =====================================================

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def classify_face_result(
    self,
    real_probability,
    fake_probability
    ):
        if fake_probability >= self.FAKE_THRESHOLD:
            return "FAKE"

        if real_probability >= self.REAL_THRESHOLD:
            return "REAL"

        return "UNCERTAIN"

    def calculate_face_risk(
    self,
    real_probability,
    fake_probability
    ):
        fake_score = fake_probability * 100

        if fake_score >= 85:
            return "VERY_HIGH"

        if fake_score >= 75:
            return "HIGH"

        if fake_score >= 60:
            return "MODERATE"

        if fake_score >= 40:
            return "LOW"

        return "VERY_LOW"
    
    # =========================================================
    # GRAD-CAM INITIALIZATION
    # =========================================================

    def _initialize_gradcam(self):

        if self.gradcam is not None:
            return

        model = model_manager.get_image_model()

        if model is None:

            raise RuntimeError(
                "Image model is not loaded."
            )

        # DeepfakeXception wrapper
        #
        # The actual Xception network
        # is inside model.model.
        #
        # conv4 was verified from the
        # model architecture.

        target_layer = model.model.conv4

        self.gradcam = GradCAM(
            model=model,
            target_layer=target_layer
        )

        print(
            "Grad-CAM initialized successfully!"
        )

        print(
            "Grad-CAM target layer: "
            "model.model.conv4"
        )

    # =========================================================
    # GENERATE GRAD-CAM
    # =========================================================

    def generate_gradcam(
        self,
        face_image: Image.Image,
        target_class: int
    ):

        self._initialize_gradcam()

        input_tensor = (
            preprocess_image(
                face_image
            ).to(
                self.device
            )
        )

        cam = self.gradcam.generate(
            input_tensor=input_tensor,
            target_class=target_class
        )

        return cam

    # =========================================================
    # ROTATE IMAGE
    # =========================================================

    def _rotate_image(
        self,
        image: np.ndarray,
        angle: float
    ) -> np.ndarray:

        h, w = image.shape[:2]

        center = (
            w // 2,
            h // 2
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR
        )

    # =========================================================
    # CONVERT ROTATED BOUNDING BOX
    # BACK TO ORIGINAL IMAGE
    # =========================================================

    def _unrotate_bbox(
        self,
        bbox: tuple,
        angle: float,
        img_width: int,
        img_height: int
    ) -> tuple:

        x, y, w, h = bbox

        center_x = img_width / 2.0
        center_y = img_height / 2.0

        corners = np.array(
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ],
            dtype=np.float32
        )

        inverse_matrix = (
            cv2.getRotationMatrix2D(
                (
                    center_x,
                    center_y
                ),
                -angle,
                1.0
            )
        )

        corners_homo = np.hstack(
            [
                corners,
                np.ones(
                    (4, 1)
                )
            ]
        )

        original_corners = np.dot(
            inverse_matrix,
            corners_homo.T
        ).T

        min_x = max(
            0,
            int(
                np.min(
                    original_corners[:, 0]
                )
            )
        )

        min_y = max(
            0,
            int(
                np.min(
                    original_corners[:, 1]
                )
            )
        )

        max_x = min(
            img_width,
            int(
                np.max(
                    original_corners[:, 0]
                )
            )
        )

        max_y = min(
            img_height,
            int(
                np.max(
                    original_corners[:, 1]
                )
            )
        )

        return (
            min_x,
            min_y,
            max_x - min_x,
            max_y - min_y
        )

    # =========================================================
    # FACE DETECTION
    # =========================================================

    def detect_face(
        self,
        file_path: Path
    ):

        # =====================================================
        # READ IMAGE
        # =====================================================

        image = cv2.imread(
            str(file_path)
        )

        if image is None:

            raise ValueError(
                "Unable to read image."
            )

        print(
            "Image shape:",
            image.shape
        )

        # =====================================================
        # 1. PRIMARY DETECTOR — YUNET
        # =====================================================

        yunet_faces = (
            yunet_detector.detect(
                image
            )
        )

        print(
            "YuNet faces detected:",
            len(yunet_faces)
        )

        # =====================================================
        # YUNET FOUND FACE
        # =====================================================

        if len(yunet_faces) > 0:

            best_face = max(
                yunet_faces,
                key=lambda face: face[-1]
            )

            x, y, width, height = (
                best_face[:4]
            )

            confidence = (
                best_face[-1]
            )

            x = int(x)
            y = int(y)
            width = int(width)
            height = int(height)

            print(
                "YuNet selected face:",
                f"x={x},",
                f"y={y},",
                f"w={width},",
                f"h={height},",
                f"confidence={confidence:.3f}"
            )

            # =================================================
            # ADD 20% MARGIN
            # =================================================

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
                image.shape[1],
                x + width + margin_x
            )

            y2 = min(
                image.shape[0],
                y + height + margin_y
            )

            # =================================================
            # CROP FACE
            # =================================================

            face_crop = image[
                y1:y2,
                x1:x2
            ]

            print(
                "YuNet face crop shape:",
                face_crop.shape
            )

            # =================================================
            # SAVE DEBUG CROP
            # =================================================

            cv2.imwrite(
                "debug_face_crop.jpg",
                face_crop
            )

            print(
                "YuNet face crop saved as "
                "debug_face_crop.jpg"
            )

            # =================================================
            # BGR → RGB
            # =================================================

            face_crop = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2RGB
            )

            return Image.fromarray(
                face_crop
            )

        # =====================================================
        # 2. YUNET FAILED
        # =====================================================

        print(
            "YuNet detected no faces."
        )

        # =====================================================
        # CHECK HAAR AVAILABILITY
        # =====================================================

        if (
            self.face_detector is None
            or self.profile_face_detector is None
        ):

            print(
                "Haar fallback is unavailable."
            )

            print(
                "No face detected."
            )

            return None

        print(
            "Falling back to Haar detection..."
        )

        # =====================================================
        # GRAYSCALE
        # =====================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # =====================================================
        # CLAHE ENHANCEMENT
        # =====================================================

        enhanced_gray = (
            self.clahe.apply(
                gray
            )
        )

        faces = []

        detection_type = "none"

        # =====================================================
        # HAAR FRONTAL DETECTION
        # =====================================================

        frontal_detected = (
            self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(40, 40)
            )
        )

        if len(
            frontal_detected
        ) > 0:

            faces = list(
                frontal_detected
            )

            detection_type = (
                "frontal"
            )

        # =====================================================
        # ENHANCED HAAR FRONTAL
        # =====================================================

        if len(faces) == 0:

            frontal_detected_enh = (
                self.face_detector.detectMultiScale(
                    enhanced_gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(40, 40)
                )
            )

            if len(
                frontal_detected_enh
            ) > 0:

                faces = list(
                    frontal_detected_enh
                )

                detection_type = (
                    "frontal_enhanced"
                )

        # =====================================================
        # HAAR PROFILE DETECTION
        # =====================================================

        if len(faces) == 0:

            rotation_angles = [
                0,
                -20,
                20
            ]

            for angle in (
                rotation_angles
            ):

                rotated_gray = (
                    self._rotate_image(
                        enhanced_gray,
                        angle
                    )
                )

                for is_flipped in (
                    False,
                    True
                ):

                    if is_flipped:

                        target = cv2.flip(
                            rotated_gray,
                            1
                        )

                    else:

                        target = (
                            rotated_gray
                        )

                    detected = (
                        self.profile_face_detector
                        .detectMultiScale(
                            target,
                            scaleFactor=1.03,
                            minNeighbors=3,
                            minSize=(35, 35)
                        )
                    )

                    if len(
                        detected
                    ) > 0:

                        img_h, img_w = (
                            target.shape[:2]
                        )

                        for (
                            bx,
                            by,
                            bw,
                            bh
                        ) in detected:

                            if is_flipped:

                                bx = (
                                    img_w
                                    - bx
                                    - bw
                                )

                            orig_bbox = (
                                self._unrotate_bbox(
                                    (
                                        bx,
                                        by,
                                        bw,
                                        bh
                                    ),
                                    angle,
                                    img_w,
                                    img_h
                                )
                            )

                            faces.append(
                                orig_bbox
                            )

                        detection_type = (
                            f"profile_rot_{angle}"
                            f"_flip_{is_flipped}"
                        )

                        break

                if len(faces) > 0:

                    break

        # =====================================================
        # CHECK HAAR RESULT
        # =====================================================

        print(
            "Fallback detection type:",
            detection_type
        )

        print(
            "Fallback faces detected:",
            len(faces)
        )

        if len(faces) == 0:

            print(
                "No face detected by "
                "YuNet or Haar."
            )

            return None

        # =====================================================
        # SELECT LARGEST HAAR FACE
        # =====================================================

        largest_face = max(
            faces,
            key=lambda face:
            face[2] * face[3]
        )

        x, y, width, height = (
            largest_face
        )

        # =====================================================
        # ADD MARGIN
        # =====================================================

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
            image.shape[1],
            x + width + margin_x
        )

        y2 = min(
            image.shape[0],
            y + height + margin_y
        )

        # =====================================================
        # CROP HAAR FACE
        # =====================================================

        face_crop = image[
            y1:y2,
            x1:x2
        ]

        print(
            "Haar fallback face crop shape:",
            face_crop.shape
        )

        # =====================================================
        # SAVE DEBUG CROP
        # =====================================================

        cv2.imwrite(
            "debug_face_crop.jpg",
            face_crop
        )

        print(
            "Haar fallback crop saved as "
            "debug_face_crop.jpg"
        )

        # =====================================================
        # BGR → RGB
        # =====================================================

        face_crop = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        return Image.fromarray(
            face_crop
        )

    # =========================================================
    # MULTI-FACE DEEPFAKE ANALYSIS
    # =========================================================

    def predict_multiple_faces(
        self,
        file_path: Path
    ):

        # =====================================================
        # 1. READ IMAGE
        # =====================================================

        image = cv2.imread(
            str(file_path)
        )

        if image is None:

            raise ValueError(
                "Unable to read image."
            )

        print(
            "Image shape:",
            image.shape
        )

        # =====================================================
        # 2. DETECT ALL FACES USING YUNET
        # =====================================================

        yunet_faces = (
            yunet_detector.detect(
                image
            )
        )

        print(
            "YuNet faces detected:",
            len(yunet_faces)
        )

        # =====================================================
        # 3. NO FACES
        # =====================================================

        if len(yunet_faces) == 0:

            return {
                "prediction": "NO_FACE",
                "confidence": 0.0,
                "face_count": 0,
                "faces": []
            }

        # =====================================================
        # 4. GET XCEPTION MODEL
        # =====================================================

        model = (
            model_manager
            .get_image_model()
        )

        if model is None:

            raise RuntimeError(
                "Image model is not loaded."
            )

        # =====================================================
        # 5. STORE INDIVIDUAL RESULTS
        # =====================================================

        face_results = []

        # =====================================================
        # 6. ANALYZE EVERY FACE
        # =====================================================

        for index, face in enumerate(
            yunet_faces,
            start=1
        ):

            # -------------------------------------------------
            # YuNet bounding box
            # -------------------------------------------------

            x = int(face[0])
            y = int(face[1])
            width = int(face[2])
            height = int(face[3])

            detection_confidence = float(
                face[-1]
            )

            print(
                f"Face {index}: "
                f"x={x}, "
                f"y={y}, "
                f"w={width}, "
                f"h={height}, "
                f"YuNet confidence="
                f"{detection_confidence:.3f}"
            )

            # -------------------------------------------------
            # Add margin
            # -------------------------------------------------

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
                image.shape[1],
                x + width + margin_x
            )

            y2 = min(
                image.shape[0],
                y + height + margin_y
            )

            # -------------------------------------------------
            # Crop face
            # -------------------------------------------------

            face_crop = image[
                y1:y2,
                x1:x2
            ]

            if face_crop.size == 0:

                print(
                    f"Face {index}: "
                    "Invalid crop. Skipping."
                )

                continue

            print(
                f"Face {index} crop shape:",
                face_crop.shape
            )

            # -------------------------------------------------
            # Save debug crop
            # -------------------------------------------------

            debug_filename = (
                f"debug_face_{index}.jpg"
            )

            cv2.imwrite(
                debug_filename,
                face_crop
            )

            # -------------------------------------------------
            # BGR → RGB
            # -------------------------------------------------

            face_crop_rgb = (
                cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )
            )

            face_pil = Image.fromarray(
                face_crop_rgb
            )

            # =================================================
            # PREPROCESS
            # =================================================

            image_tensor = (
                preprocess_image(
                    face_pil
                ).to(
                    self.device
                )
            )

            print(
                "Input tensor shape:",
                image_tensor.shape
            )

            # =================================================
            # XCEPTION INFERENCE
            # =================================================

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

            # =================================================
            # GET PROBABILITIES
            # =================================================

            real_probability = (
                probabilities[0][0].item()
            )

            fake_probability = (
                probabilities[0][1].item()
            )

            # =================================================
            # DETERMINE RESULT
            # =================================================
            #
            # Strong evidence is required before calling a face
            # FAKE or REAL. Predictions in the middle are marked
            # UNCERTAIN instead of forcing a binary decision.
            #
            if fake_probability >= self.FAKE_THRESHOLD:

                prediction = "FAKE"
                confidence = fake_probability

            elif real_probability >= self.REAL_THRESHOLD:

                prediction = "REAL"
                confidence = real_probability

            else:

                prediction = "UNCERTAIN"
                confidence = max(
                    fake_probability,
                    real_probability
                )

            confidence_percentage = float(
                confidence * 100
            )

            # =================================================
            # GRAD-CAM TARGET CLASS
            # =================================================

            target_class = (
                1
                if prediction == "FAKE"
                else 0
            )

            # =================================================
            # GENERATE GRAD-CAM
            # =================================================

            try:

                cam = self.generate_gradcam(
                    face_image=face_pil,
                    target_class=target_class
                )

                print(
                    f"Face {index}: "
                    "Grad-CAM generated successfully."
                )

            except Exception as e:

                print(
                    f"Face {index}: "
                    f"Grad-CAM failed: {e}"
                )

                cam = None

            # =================================================
            # SAVE GRAD-CAM OVERLAY
            # =================================================

            gradcam_path = None

            if cam is not None:

                try:

                    overlay = self.gradcam.overlay(
                        face_crop,
                        cam,
                        alpha=0.45
                    )

                    # ---------------------------------------------
                    # Create Grad-CAM output directory
                    # ---------------------------------------------

                    gradcam_dir = Path(
                        "outputs"
                    ) / "gradcam"

                    gradcam_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    # ---------------------------------------------
                    # Unique filename for this face
                    # ---------------------------------------------

                    gradcam_filename = (
                        f"face_{index}_gradcam.jpg"
                    )

                    gradcam_file = (
                        gradcam_dir /
                        gradcam_filename
                    )

                    # ---------------------------------------------
                    # Save image
                    # ---------------------------------------------

                    success = cv2.imwrite(
                        str(gradcam_file),
                        overlay
                    )

                    if not success:

                        raise RuntimeError(
                            "cv2.imwrite() failed."
                        )

                    gradcam_path = str(
                        gradcam_file
                    ).replace("\\", "/")

                    print(
                        f"Face {index}: "
                        f"Grad-CAM saved to "
                        f"{gradcam_path}"
                    )

                except Exception as e:

                    print(
                        f"Face {index}: "
                        f"Could not save Grad-CAM "
                        f"overlay: {e}"
                    )

                    gradcam_path = None
            # =================================================
            # RISK LEVEL
            # =================================================

            risk_level = calculate_risk_level(
                prediction,
                confidence_percentage
            )

            # =================================================
            # STORE RESULT
            # =================================================

            result = {

                "face_id": index,

                "x": x,
                "y": y,

                "width": width,
                "height": height,

                "detection_confidence": round(
                    float(
                        detection_confidence
                    ) * 100,
                    2
                ),

                "prediction": prediction,

                "confidence": round(
                    float(confidence) * 100,
                    2
                ),

                "real_probability": round(
                    float(real_probability) * 100,
                    2
                ),

                "fake_probability": round(
                    float(fake_probability) * 100,
                    2
                ),

                "risk_level": risk_level,

                "gradcam_generated": (
                    cam is not None
                ),

                "gradcam_path": 
                    gradcam_path,
            }

            face_results.append(
                result
            )

            print(
                f"Face {index} result: "
                f"{prediction} "
                f"({confidence * 100:.2f}%)"
            )

        # =====================================================
        # 7. IF ALL CROPS FAILED
        # =====================================================

        if len(face_results) == 0:

            return {
                "prediction": "NO_FACE",
                "confidence": 0.0,
                "face_count": 0,
                "faces": []
            }

        # =====================================================
        # 8. CALCULATE OVERALL RESULT
        # =====================================================

        fake_faces = [
            face
            for face in face_results
            if face["prediction"] == "FAKE"
        ]

        real_faces = [
            face
            for face in face_results
            if face["prediction"] == "REAL"
        ]

        uncertain_faces = [
            face
            for face in face_results
            if face["prediction"] == "UNCERTAIN"
        ]

        strong_fake_faces = [
        face
        for face in fake_faces
        if face["fake_probability"] >= 75.0
        ]

        fake_count = len(fake_faces)
        real_count = len(real_faces)
        uncertain_count = len(uncertain_faces)
        strong_fake_count = len(strong_fake_faces)

        overall_risk = calculate_overall_risk(
        face_results
        )   

        # -----------------------------------------------------
        # Strong fake evidence
        # -----------------------------------------------------
        # Only a face with >= 75% fake probability is considered
        # strong enough to make the whole image LIKELY_FAKE.

        strong_fake_faces = [
            face
            for face in fake_faces
            if face["fake_probability"] >= 75.0
        ]

        fake_count = len(fake_faces)
        real_count = len(real_faces)
        uncertain_count = len(uncertain_faces)
        strong_fake_count = len(strong_fake_faces)

        # =====================================================
        # OVERALL RISK ENGINE
        # =====================================================

        overall_risk = calculate_overall_risk(
            face_results
        )

        print(
            "Overall risk score:",
            overall_risk["risk_score"]
        )

        print(
            "Overall risk level:",
            overall_risk["risk_level"]
        )

        print(
            "Risk reason:",
            overall_risk["reason"]
        )

        # =====================================================
        # OVERALL INTERPRETATION
        # =====================================================

        if strong_fake_count > 0:

            overall_prediction = "LIKELY_FAKE"

        elif uncertain_count > 0:

            overall_prediction = "UNCERTAIN"

        else:

            overall_prediction = "LIKELY_REAL"

        print("========================================")
        print("OVERALL IMAGE ANALYSIS")
        print("========================================")
        print("Total faces:", len(face_results))
        print("Real faces:", real_count)
        print("Fake faces:", fake_count)
        print("Uncertain faces:", uncertain_count)
        print("Strong fake faces:", strong_fake_count)
        print("Overall result:", overall_prediction)
        print("Overall risk:", overall_risk["risk_level"])
        print("========================================")

        # -----------------------------------------------------
        # Overall confidence
        # -----------------------------------------------------

        overall_confidence = float(
            max(
                face["confidence"]
                for face in face_results
            )
        )

        # =====================================================
        # 9. FINAL RESULT
        # =====================================================

        result = {

            "prediction":
                overall_prediction,

            "confidence":
                round(
                    float(
                        overall_confidence
                    ),
                    2
                ),

            # =============================================
            # Overall Risk
            # =============================================

            "risk_score":
                overall_risk["risk_score"],

            "risk_level":
                overall_risk["risk_level"],

            "risk_reason":
                overall_risk["reason"],

            # =============================================
            # Face Statistics
            # =============================================

            "face_count":
                len(face_results),

            "fake_face_count":
                len(fake_faces),

            "real_face_count":
                len(real_faces),

            "uncertain_face_count":
                len(uncertain_faces),

            "strong_fake_face_count":
                len(strong_fake_faces),

            # =============================================
            # Individual Face Results
            # =============================================

            "faces":
                face_results
        }

        return result

    # =========================================================
    # SINGLE-FACE DEEPFAKE PREDICTION
    # =========================================================

    def predict(
        self,
        file_path: Path
    ):

        # =====================================================
        # 1. DETECT FACE
        # =====================================================

        face = self.detect_face(
            file_path
        )

        # =====================================================
        # NO FACE
        # =====================================================

        if face is None:

            return {
                "prediction": "NO_FACE",
                "confidence": 0.0,
                "real_probability": 0.0,
                "fake_probability": 0.0,
                "face_detected": False
            }

        # =====================================================
        # 2. PREPROCESS FACE
        # =====================================================

        image_tensor = (
            preprocess_image(
                face
            ).to(
                self.device
            )
        )

        print(
            "Input tensor shape:",
            image_tensor.shape
        )

        # =====================================================
        # 3. GET XCEPTION MODEL
        # =====================================================

        model = (
            model_manager
            .get_image_model()
        )

        if model is None:

            raise RuntimeError(
                "Image model is not loaded."
            )

        # =====================================================
        # 4. RUN INFERENCE
        # =====================================================

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

        # =====================================================
        # 5. GET PROBABILITIES
        # =====================================================

        real_probability = (
            probabilities[0][0].item()
        )

        fake_probability = (
            probabilities[0][1].item()
        )

        # =====================================================
        # 6. DETERMINE PREDICTION
        # =====================================================

        if fake_probability >= self.FAKE_THRESHOLD:

            prediction = "FAKE"
            confidence = fake_probability

        elif real_probability >= self.REAL_THRESHOLD:

            prediction = "REAL"
            confidence = real_probability

        else:

            prediction = "UNCERTAIN"
            confidence = max(
                fake_probability,
                real_probability
            )

        # =====================================================
        # 7. RETURN RESULT
        # =====================================================

        return {

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
                ),

            "face_detected":
                True
        }


# =============================================================
# GLOBAL IMAGE DETECTOR
# =============================================================

image_detector = ImageDetector()