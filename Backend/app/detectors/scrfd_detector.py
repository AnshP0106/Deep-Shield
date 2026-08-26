from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import onnxruntime as ort


def distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape: Tuple[int, int] = None) -> np.ndarray:
    """Decode distance predictions (left, top, right, bottom) from anchor points to [x1, y1, x2, y2]."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


class SCRFDDetector:
    """
    SCRFD ONNX face detector using InsightFace architecture and ONNX Runtime CPU inference.
    """

    def __init__(
        self,
        model_path: str = "models/weights/scrfd_10g_bnkps.onnx",
        conf_threshold: float = 0.50,
        nms_threshold: float = 0.40,
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SCRFD model weight file not found at: {self.model_path.resolve()}")

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Normalization constants (standard InsightFace configuration)
        self.input_mean = 127.5
        self.input_std = 128.0

        # Feature map pyramid configuration
        self._feat_stride_fpn = [8, 16, 32]
        self.fmc = len(self._feat_stride_fpn)
        self.center_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}

        # ONNX Runtime CPU Session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Automatically determine anchors per stride based on outputs count
        if len(self.output_names) == 6:
            self._num_anchors = 1
            self.use_kps = False
        elif len(self.output_names) == 9:
            self._num_anchors = 2
            self.use_kps = False
        elif len(self.output_names) == 10:
            self._num_anchors = 1
            self.use_kps = True
        elif len(self.output_names) == 15:
            self._num_anchors = 2
            self.use_kps = True
        else:
            self._num_anchors = 2
            self.use_kps = len(self.output_names) % 3 == 0

    def _generate_anchor_centers(self, height: int, width: int, stride: int, num_anchors: int) -> np.ndarray:
        key = (height, width, stride, num_anchors)
        if key in self.center_cache:
            return self.center_cache[key]

        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))

        if len(self.center_cache) < 100:
            self.center_cache[key] = anchor_centers
        return anchor_centers

    @staticmethod
    def _nms(dets: np.ndarray, thresh: float) -> List[int]:
        """Perform Non-Maximum Suppression on [x1, y1, x2, y2, score]."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Accepts an OpenCV BGR image and returns a list of detected faces formatted as:
        [
            {
                "bbox": [x, y, w, h],
                "confidence": float
            }
        ]
        """
        if image is None or image.size == 0:
            return []

        img_h, img_w = image.shape[:2]

        # Compute scaling factor preserving aspect ratio (or default fixed size)
        target_w, target_h = self.input_size
        im_ratio = float(img_h) / float(img_w)
        model_ratio = float(target_h) / float(target_w)

        if im_ratio > model_ratio:
            new_h = target_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = target_w
            new_h = int(new_w * im_ratio)

        det_scale = float(new_h) / float(img_h)
        resized_img = cv2.resize(image, (new_w, new_h))

        # Pad into uniform canvas
        det_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized_img

        # Normalize to NCHW input blob
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )

        outputs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            score_out = outputs[idx]
            bbox_out = outputs[idx + self.fmc] * stride

            feat_h = target_h // stride
            feat_w = target_w // stride
            num_anchors = self._num_anchors

            score = score_out.reshape((-1, 1))
            bbox = bbox_out.reshape((-1, 4))

            anchor_centers = self._generate_anchor_centers(feat_h, feat_w, stride, num_anchors)

            pos_inds = np.where(score >= self.conf_threshold)[0]
            if len(pos_inds) == 0:
                continue

            pos_scores = score[pos_inds]
            pos_bbox = bbox[pos_inds]
            pos_anchors = anchor_centers[pos_inds]

            decoded_bboxes = distance2bbox(pos_anchors, pos_bbox)
            decoded_bboxes = decoded_bboxes / det_scale

            scores_list.append(pos_scores)
            bboxes_list.append(decoded_bboxes)

        if not scores_list:
            return []

        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list)

        dets = np.hstack((bboxes, scores))
        keep_indices = self._nms(dets, self.nms_threshold)
        dets = dets[keep_indices]

        results = []
        for det in dets:
            x1, y1, x2, y2, score = det

            # Clamp coordinates to original image bounds
            x1_clamped = max(0, min(int(round(x1)), img_w))
            y1_clamped = max(0, min(int(round(y1)), img_h))
            x2_clamped = max(0, min(int(round(x2)), img_w))
            y2_clamped = max(0, min(int(round(y2)), img_h))

            w_clamped = max(0, x2_clamped - x1_clamped)
            h_clamped = max(0, y2_clamped - y1_clamped)

            if w_clamped <= 0 or h_clamped <= 0:
                continue

            results.append({
                "bbox": [x1_clamped, y1_clamped, w_clamped, h_clamped],
                "confidence": round(float(score), 4)
            })

        return results


scrfd_detector = SCRFDDetector()