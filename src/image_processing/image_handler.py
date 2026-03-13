"""
Image processing for fire detection
- Frame extraction from drone video stream
- Preprocessing (resizing, normalization)
- Smoke and flame enhancement
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageHandler:
    """Handles image processing from drone video stream"""
    
    def __init__(self, config: dict):
        """Initialize with config"""
        self.config = config
        self.img_size = 416  # Standard size for detection
        
    def extract_frames(self, video_path: str, sample_rate: int = 30) -> list:
        """
        Extract frames from drone video
        Args:
            video_path: path to drone video file
            sample_rate: extract every Nth frame
        Returns:
            list of numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                frames.append(frame)
                logger.debug(f"Extracted frame {frame_count}")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection
        Args:
            frame: input image array
        Returns:
            preprocessed image
        """
        # Resize to standard size
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def enhance_fire_colors(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance red/orange colors for better fire detection
        Args:
            frame: input image
        Returns:
            enhanced image
        """
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red/orange colors (fire)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        fire_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply mask to original image
        enhanced = cv2.bitwise_and(frame, frame, mask=fire_mask)
        
        return enhanced
    
    def detect_smoke(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect smoke regions (grayish colors)
        Args:
            frame: input image
        Returns:
            smoke mask
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Smoke is low saturation with medium-high value
        lower_smoke = np.array([0, 0, 50])
        upper_smoke = np.array([180, 50, 200])
        
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        
        return smoke_mask
    
    def save_frame(self, frame: np.ndarray, output_path: str):
        """Save frame to disk"""
        cv2.imwrite(output_path, frame)
        logger.debug(f"Saved frame to {output_path}")
