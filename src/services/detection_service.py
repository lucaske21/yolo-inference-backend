"""
Detection service module for object detection operations.

This module provides the DetectionService class that encapsulates
all detection and inference logic for YOLO models.
"""

import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from logger import get_logger


logger = get_logger(__name__)


class DetectionResult:
    """
    Data class representing a single detection result.
    
    Attributes:
        class_id (int): Detected class ID
        class_name (str): Detected class name
        confidence (float): Detection confidence score
        x1, y1, x2, y2 (int): Bounding box coordinates
    """
    
    def __init__(
        self,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }


class DetectionService:
    """
    Service class for performing object detection using YOLO models.
    
    This class encapsulates all detection logic including image processing,
    model inference, result processing, and output image generation.
    
    Attributes:
        inference_sessions: Model inference session manager
        conf_thres (float): Confidence threshold for detection
        iou_thres (float): IoU threshold for detection
        output_img_base_path (str): Path to save output images
    """
    
    def __init__(
        self,
        inference_sessions,
        conf_thres: float,
        iou_thres: float,
        output_img_base_path: str
    ):
        """
        Initialize the DetectionService.
        
        Args:
            inference_sessions: InferenceSessions manager instance
            conf_thres: Confidence threshold for detection
            iou_thres: IoU threshold for detection
            output_img_base_path: Path to save output images
        """
        self.inference_sessions = inference_sessions
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.output_img_base_path = output_img_base_path
        logger.info(
            f"DetectionService initialized with conf_thres={conf_thres}, "
            f"iou_thres={iou_thres}"
        )
    
    def decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode image bytes to numpy array.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Decoded image as numpy array, or None if decoding fails
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error(
                    f"Failed to decode image: size={len(image_bytes)} bytes, "
                    f"may be invalid format or corrupted"
                )
                return None
            logger.debug(f"Image decoded successfully with shape: {img.shape}")
            return img
        except Exception as e:
            logger.error(
                f"Error decoding image: {e}, size={len(image_bytes)} bytes"
            )
            return None
    
    def run_inference(
        self,
        image: np.ndarray,
        model_id: int
    ) -> Optional[Any]:
        """
        Run inference on an image using specified model.
        
        Args:
            image: Input image as numpy array
            model_id: ID of the model to use
            
        Returns:
            Inference results or None if inference fails
        """
        try:
            model = self.inference_sessions.get_session(model_id)
            if model is None:
                logger.error(f"Model ID {model_id} not found")
                return None
            
            logger.info(f"Running inference with model ID {model_id}")
            results = model.predict(
                source=image,
                conf=self.conf_thres,
                iou=self.iou_thres,
                show_labels=True,
                show_conf=True
            )
            logger.debug(f"Inference completed for model ID {model_id}")
            return results
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return None
    
    def process_results(
        self,
        results: Any,
        model_id: int
    ) -> List[DetectionResult]:
        """
        Process inference results into structured format.
        
        Args:
            results: Raw inference results from YOLO model
            model_id: ID of the model used
            
        Returns:
            List of DetectionResult objects
        """
        detections = []
        model_labels = self.inference_sessions.get_label_names(model_id)
        
        if model_labels is None:
            logger.error(f"Labels for Model ID {model_id} not found")
            return detections
        
        try:
            for result in results:
                for box in result.boxes.data.tolist():
                    detection = DetectionResult(
                        class_id=int(box[5]),
                        class_name=model_labels[int(box[5])],
                        confidence=float(box[4]),
                        x1=int(box[0]),
                        y1=int(box[1]),
                        x2=int(box[2]),
                        y2=int(box[3])
                    )
                    detections.append(detection)
            
            logger.info(f"Processed {len(detections)} detections")
        except Exception as e:
            logger.error(f"Error processing results: {e}")
        
        return detections
    
    def save_output_image(
        self,
        results: Any,
        filename: str
    ) -> bool:
        """
        Save annotated image with detection results.
        
        Args:
            results: Inference results with plotted boxes
            filename: Output filename
            
        Returns:
            bool: True if image saved successfully
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_img_base_path, exist_ok=True)
            
            # Draw boxes and labels on the image
            result_img = results[0].plot()
            
            # Save image
            output_path = os.path.join(self.output_img_base_path, filename)
            cv2.imwrite(output_path, result_img)
            
            logger.info(f"Output image saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving output image: {e}")
            return False
    
    def detect_objects(
        self,
        image_bytes: bytes,
        model_id: int,
        filename: str
    ) -> Dict[str, Any]:
        """
        Perform complete object detection pipeline.
        
        Args:
            image_bytes: Raw image bytes
            model_id: ID of the model to use
            filename: Original filename for saving output
            
        Returns:
            Dictionary with predictions or error message
        """
        logger.info(f"Starting detection for model_id={model_id}, filename={filename}")
        
        # Decode image
        img = self.decode_image(image_bytes)
        if img is None:
            return {"error": "Failed to decode image"}
        
        # Check if model exists
        if self.inference_sessions.get_session(model_id) is None:
            error_msg = f"Model ID {model_id} not found"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Run inference
        results = self.run_inference(img, model_id)
        if results is None:
            return {"error": "Inference failed"}
        
        # Process results
        detections = self.process_results(results, model_id)
        
        # Save output image
        self.save_output_image(results, filename)
        
        # Convert to dictionary format
        predictions_dict = [det.to_dict() for det in detections]
        
        logger.info(f"Detection completed with {len(predictions_dict)} objects detected")
        return {"predictions": predictions_dict}
