"""
Fire detection using deep learning
- YOLOv8 or similar for real-time detection
- Flame and smoke classification
- Confidence filtering
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FireDetector:
    """Fire detection using pre-trained neural network"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize fire detector
        Args:
            model_path: path to trained model weights
            confidence_threshold: minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained fire detection model"""
        try:
            # Try YOLOv8 first (modern, recommended)
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"Loaded YOLOv8 model from {self.model_path}")
        except ImportError:
            try:
                # Fallback to TensorFlow
                import tensorflow as tf
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded TensorFlow model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect fire in frame
        Args:
            frame: input image (BGR)
        Returns:
            list of detections with format:
            {
                'class': 'fire'|'smoke',
                'confidence': float,
                'bbox': (x1, y1, x2, y2),
                'area_pixels': int
            }
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        detections = []
        
        try:
            results = self.model(frame)
            
            # Parse results based on model type
            for detection in results:
                if hasattr(detection, 'boxes'):  # YOLOv8
                    for box in detection.boxes:
                        conf = float(box.conf[0])
                        if conf >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0])
                            
                            detections.append({
                                'class': 'fire' if class_id == 0 else 'smoke',
                                'confidence': conf,
                                'bbox': (x1, y1, x2, y2),
                                'area_pixels': (x2 - x1) * (y2 - y1),
                            })
            
            logger.debug(f"Detected {len(detections)} fire/smoke regions")
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        return detections
    
    def filter_detections(self, detections: List[dict], min_area_pixels: int = 100) -> List[dict]:
        """
        Filter detections by area and confidence
        Args:
            detections: raw detections
            min_area_pixels: minimum bounding box area
        Returns:
            filtered detections
        """
        filtered = [d for d in detections if d['area_pixels'] >= min_area_pixels]
        logger.debug(f"Filtered detections: {len(detections)} -> {len(filtered)}")
        return filtered
    
    def get_detection_stats(self, detections: List[dict]) -> dict:
        """Get statistics from detections"""
        if not detections:
            return {'count': 0, 'avg_confidence': 0, 'fire_count': 0, 'smoke_count': 0}
        
        fire_detections = [d for d in detections if d['class'] == 'fire']
        smoke_detections = [d for d in detections if d['class'] == 'smoke']
        
        return {
            'count': len(detections),
            'fire_count': len(fire_detections),
            'smoke_count': len(smoke_detections),
            'avg_confidence': np.mean([d['confidence'] for d in detections]),
            'max_confidence': max([d['confidence'] for d in detections]),
        }


class DetectionTracker:
    """Track detections across frames for temporal consistency"""
    
    def __init__(self, max_history: int = 5):
        """
        Initialize tracker
        Args:
            max_history: number of frames to track
        """
        self.max_history = max_history
        self.detection_history = []
    
    def add_detections(self, frame_id: int, detections: List[dict]):
        """Add detections from a frame"""
        self.detection_history.append({
            'frame_id': frame_id,
            'detections': detections,
            'timestamp': None,
        })
        
        # Keep only recent frames
        if len(self.detection_history) > self.max_history:
            self.detection_history = self.detection_history[-self.max_history:]
    
    def get_persistent_detections(self, min_frames: int = 2) -> List[dict]:
        """
        Get detections that appeared in multiple frames
        Args:
            min_frames: minimum frames to be considered persistent
        Returns:
            persistent detections
        """
        # Simple approach: detections appearing in recent frames
        if len(self.detection_history) < min_frames:
            return []
        
        # TODO: Implement proper spatial tracking/association
        recent_detections = self.detection_history[-min_frames:]
        all_detections = []
        for frame_data in recent_detections:
            all_detections.extend(frame_data['detections'])
        
        return all_detections
