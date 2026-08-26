from typing import List, Dict, Any, Tuple
import numpy as np


class TrackedFace:
    """Represents a single face track across video frames."""
    def __init__(self, track_id: int, bbox: List[int], score: float, frame_idx: int, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.score = score
        self.last_frame_idx = frame_idx
        self.last_timestamp = timestamp
        self.hits = 1
        self.time_since_update = 0


class IoUFaceTracker:
    """
    Lightweight, persistent face tracker using Intersection over Union (IoU).
    Does not require deep appearance embeddings; optimized for sampled video frames.
    """
    def __init__(self, iou_threshold: float = 0.35, max_lost_frames: int = 5):
        """
        Args:
            iou_threshold: Minimum IoU required to associate a detection with an existing track.
            max_lost_frames: Number of consecutive frames a face can be undetected before track termination.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 1
        self.tracks: List[TrackedFace] = []

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """Computes IoU between two [x, y, w, h] bounding boxes."""
        ax1, ay1, aw, ah = box_a
        ax2, ay2 = ax1 + aw, ay1 + ah

        bx1, by1, bw, bh = box_b
        bx2, by2 = bx1 + bw, by1 + bh

        # Determine the coordinates of the intersection rectangle
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Compute area of both bounding boxes
        area_a = aw * ah
        area_b = bw * bh
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Updates active tracks with detections from the current frame.

        Args:
            detections: List of dicts, each containing:
                        - 'bbox': [x, y, w, h]
                        - 'confidence': float
            frame_idx: Integer index of the current frame.
            timestamp: Timestamp of the current frame (in seconds).

        Returns:
            List of tracked face objects for the current frame:
            [
                {
                    "face_id": int,
                    "bbox": [x, y, w, h],
                    "confidence": float,
                    "timestamp": float,
                    "frame_idx": int
                }, ...
            ]
        """
        for track in self.tracks:
            track.time_since_update += 1

        matched_track_indices = set()
        matched_det_indices = set()
        active_tracked_results = []

        if len(self.tracks) > 0 and len(detections) > 0:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = self._compute_iou(track.bbox, det["bbox"])

            # Greedy bipartite matching based on IoU threshold
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0
                if max_iou < self.iou_threshold:
                    break

                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                # Update track state
                track = self.tracks[t_idx]
                det = detections[d_idx]
                track.bbox = det["bbox"]
                track.score = det["confidence"]
                track.last_frame_idx = frame_idx
                track.last_timestamp = timestamp
                track.hits += 1
                track.time_since_update = 0

                matched_track_indices.add(t_idx)
                matched_det_indices.add(d_idx)

                active_tracked_results.append({
                    "face_id": track.track_id,
                    "bbox": track.bbox,
                    "confidence": track.score,
                    "timestamp": timestamp,
                    "frame_idx": frame_idx
                })

                # Invalidate matched row and column
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                new_track = TrackedFace(
                    track_id=self.next_track_id,
                    bbox=det["bbox"],
                    score=det["confidence"],
                    frame_idx=frame_idx,
                    timestamp=timestamp
                )
                self.tracks.append(new_track)
                self.next_track_id += 1

                active_tracked_results.append({
                    "face_id": new_track.track_id,
                    "bbox": new_track.bbox,
                    "confidence": new_track.score,
                    "timestamp": timestamp,
                    "frame_idx": frame_idx
                })

        # Remove dead tracks that exceeded the allowable lost frame budget
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_lost_frames
        ]

        return active_tracked_results