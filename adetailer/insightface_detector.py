"""
InsightFace-based face detection for enhanced accuracy.
Provides high-precision face detection as an alternative to MediaPipe.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError as e:
    INSIGHTFACE_AVAILABLE = False
    logger.warning(f"InsightFace not available: {e}")
except Exception as e:
    INSIGHTFACE_AVAILABLE = False
    logger.warning(f"InsightFace import error: {e}")


class InsightFaceDetector:
    """
    InsightFace-based face detector for enhanced accuracy.
    Provides better face detection than YOLO models for small or angled faces.
    """
    
    def __init__(self, model_name: str = "buffalo_l"):
        """
        Initialize InsightFace detector.
        
        Parameters
        ----------
        model_name : str
            InsightFace model name. Options: 'buffalo_l', 'buffalo_m', 'buffalo_s'
        """
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace is not available. Install with: pip install insightface")
        
        self.model_name = model_name
        self.app = None
        self._initialize_app()
    
    def _initialize_app(self):
        """Initialize InsightFace FaceAnalysis app."""
        try:
            self.app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"InsightFace detector initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise
    
    def detect_faces(self, image: Image.Image, confidence: float = 0.5) -> list[dict[str, Any]]:
        """
        Detect faces in image using InsightFace.
        
        Parameters
        ----------
        image : Image.Image
            Input image
        confidence : float
            Detection confidence threshold (0.0-1.0)
            
        Returns
        -------
        list[dict[str, Any]]
            List of detected faces with bounding boxes and confidence scores
        """
        if self.app is None:
            raise RuntimeError("InsightFace app not initialized")
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if img_array.shape[2] == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = self.app.get(img_array)
        
        # Filter by confidence and format results
        results = []
        for face in faces:
            if face.det_score >= confidence:
                bbox = face.bbox.astype(int)
                results.append({
                    'bbox': bbox,
                    'confidence': float(face.det_score),
                    'landmarks': face.kps if hasattr(face, 'kps') else None,
                    'embedding': face.embedding if hasattr(face, 'embedding') else None
                })
        
        logger.debug(f"InsightFace detected {len(results)} faces with confidence >= {confidence}")
        return results
    
    def get_face_boxes(self, image: Image.Image, confidence: float = 0.5) -> list[list[int]]:
        """
        Get face bounding boxes in format compatible with ADetailer.
        
        Parameters
        ----------
        image : Image.Image
            Input image
        confidence : float
            Detection confidence threshold
            
        Returns
        -------
        list[list[int]]
            List of bounding boxes [x1, y1, x2, y2]
        """
        faces = self.detect_faces(image, confidence)
        return [face['bbox'].tolist() for face in faces]


def create_insightface_detector(model_name: str = "buffalo_l") -> InsightFaceDetector | None:
    """
    Create InsightFace detector instance.
    
    Parameters
    ----------
    model_name : str
        InsightFace model name
        
    Returns
    -------
    InsightFaceDetector | None
        Detector instance or None if InsightFace is not available
    """
    if not INSIGHTFACE_AVAILABLE:
        logger.warning("InsightFace not available")
        return None
    
    try:
        return InsightFaceDetector(model_name)
    except Exception as e:
        logger.error(f"Failed to create InsightFace detector: {e}")
        return None


# Global detector instance
_insightface_detector = None

def get_insightface_detector() -> InsightFaceDetector | None:
    """Get global InsightFace detector instance."""
    global _insightface_detector
    if _insightface_detector is None:
        _insightface_detector = create_insightface_detector()
    return _insightface_detector
