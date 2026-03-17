"""
Fire detection using deep learning
Object detection utilities.
- YOLOv8 model loading and inference
- Class filtering for placeholder object models like cars
- Lightweight track association across frames for triangulation
"""

import numpy as np
from typing import Dict, Iterable, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass, field


def _normalize_target_classes(target_classes: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not target_classes:
        return None
    normalized = {str(item).strip().lower() for item in target_classes if str(item).strip()}
    return normalized or None


class YoloObjectDetector:
    """YOLO-based object detector that can be configured for any target class set."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        target_classes: Optional[Iterable[str]] = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = _normalize_target_classes(target_classes)
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLOv8 model from {self.model_path}")
        except ImportError:
            logger.error("ultralytics is not installed. Install it to use YOLOv8 detection.")
            self.model = None
        except Exception as exc:
            logger.error(f"Failed to load YOLOv8 model: {exc}")
            self.model = None

    @staticmethod
    def _result_names(result) -> Dict[int, str]:
        names = getattr(result, 'names', None)
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    def detect(self, frame: np.ndarray) -> List[dict]:
        if self.model is None:
            logger.error("Model not loaded")
            return []

        detections: List[dict] = []
        try:
            results = self.model.predict(frame, verbose=False)
            for result in results:
                if not hasattr(result, 'boxes'):
                    continue

                names = self._result_names(result)
                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    class_name = names.get(class_id, str(class_id)).lower()

                    if self.target_classes and class_name not in self.target_classes:
                        continue

                    detections.append(
                        {
                            'class': class_name,
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'area_pixels': max(0, x2 - x1) * max(0, y2 - y1),
                        }
                    )

            logger.debug(f"Detected {len(detections)} objects")
        except Exception as exc:
            logger.error(f"Detection failed: {exc}")
        return detections

    def filter_detections(self, detections: List[dict], min_area_pixels: int = 100) -> List[dict]:
        filtered = [item for item in detections if item['area_pixels'] >= min_area_pixels]
        logger.debug(f"Filtered detections: {len(detections)} -> {len(filtered)}")
        return filtered

    def get_detection_stats(self, detections: List[dict]) -> dict:
        if not detections:
            return {'count': 0, 'avg_confidence': 0.0, 'max_confidence': 0.0, 'classes': {}}

        classes: Dict[str, int] = {}
        for detection in detections:
            classes[detection['class']] = classes.get(detection['class'], 0) + 1

        return {
            'count': len(detections),
            'avg_confidence': float(np.mean([item['confidence'] for item in detections])),
            'max_confidence': float(max(item['confidence'] for item in detections)),
            'classes': classes,
        }


class FireDetector(YoloObjectDetector):
    """Backward-compatible fire detector wrapper."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            target_classes=['fire', 'smoke'],
        )


@dataclass
class TrackState:
    track_id: int
    class_name: str
    detections: List[dict] = field(default_factory=list)
    last_frame_index: int = -1


class DetectionTracker:
    """Associate detections across nearby frames with simple IoU and center distance matching."""

    def __init__(
        self,
        max_frame_gap: int = 3,
        min_iou: float = 0.05,
        max_normalized_distance: float = 0.18,
    ):
        self.max_frame_gap = max_frame_gap
        self.min_iou = min_iou
        self.max_normalized_distance = max_normalized_distance

    @staticmethod
    def _bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        if intersection <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def _normalized_center_distance(detection: dict, track_detection: dict) -> float:
        box_a = detection['bbox']
        box_b = track_detection['bbox']
        center_ax = (box_a[0] + box_a[2]) / 2.0
        center_ay = (box_a[1] + box_a[3]) / 2.0
        center_bx = (box_b[0] + box_b[2]) / 2.0
        center_by = (box_b[1] + box_b[3]) / 2.0
        frame_w = float(detection['frame_width'])
        frame_h = float(detection['frame_height'])
        dx = (center_ax - center_bx) / max(frame_w, 1.0)
        dy = (center_ay - center_by) / max(frame_h, 1.0)
        return float((dx ** 2 + dy ** 2) ** 0.5)

    def build_tracks(self, frame_detections: List[List[dict]]) -> List[dict]:
        active_tracks: List[TrackState] = []
        completed_tracks: List[TrackState] = []
        next_track_id = 1

        for frame_index, detections in enumerate(frame_detections):
            detections = detections or []
            used_tracks: set[int] = set()
            for detection in detections:
                best_track: Optional[TrackState] = None
                best_score = (-1.0, float('inf'))
                for track in active_tracks:
                    if track.class_name != detection['class']:
                        continue
                    if track.track_id in used_tracks:
                        continue
                    if frame_index - track.last_frame_index > self.max_frame_gap:
                        continue

                    previous = track.detections[-1]
                    iou = self._bbox_iou(detection['bbox'], previous['bbox'])
                    distance = self._normalized_center_distance(detection, previous)
                    if iou < self.min_iou and distance > self.max_normalized_distance:
                        continue

                    score = (iou, -distance)
                    if score > best_score:
                        best_score = score
                        best_track = track

                if best_track is None:
                    best_track = TrackState(track_id=next_track_id, class_name=detection['class'])
                    next_track_id += 1
                    active_tracks.append(best_track)

                detection['track_id'] = best_track.track_id
                best_track.detections.append(detection)
                best_track.last_frame_index = frame_index
                used_tracks.add(best_track.track_id)

            still_active: List[TrackState] = []
            for track in active_tracks:
                if frame_index - track.last_frame_index > self.max_frame_gap:
                    completed_tracks.append(track)
                else:
                    still_active.append(track)
            active_tracks = still_active

        completed_tracks.extend(active_tracks)
        return [
            {
                'track_id': track.track_id,
                'class': track.class_name,
                'detections': track.detections,
            }
            for track in completed_tracks
        ]
