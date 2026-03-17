import cv2
import mediapipe as mp
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union
from inference.mediapipe.base_mediapipe import BaseMediaPipeModel

class FaceDetectMediapipeModel(BaseMediaPipeModel):

    def __init__(self, model_name: str = "mediapipe_face_detect",
        device: str = "cpu",
        min_detection_confidence: float = 0.9
    ) -> None:
        super().__init__(model_name, device)
        self._min_detection_confidence = min_detection_confidence

    def load(self) -> None:
        self._model = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self._min_detection_confidence
        )
        self._is_loaded = True

    def predict(self, frame: Union[str, np.ndarray], inputType: str = "frame") -> Dict[int, Dict[str, Any]]:
        """
        Predict with support for:
        - inputType="file": frame must be str (file path)
        - inputType="frame": frame must be np.ndarray (crop frame)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self._model is None:
            raise RuntimeError("Model not initialized")

        if inputType == "file":
            if not isinstance(frame, str):
                raise TypeError(f"Expected str when inputType='file', got {type(frame)}")
            frame = cv2.imread(frame)
            if frame is None:
                raise ValueError(f"Could not load frame from path: {frame}")
        elif inputType == "frame":
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"Expected np.ndarray when inputType='frame', got {type(frame)}")
            frame = frame
        else:
            raise ValueError(f"Invalid inputType: {inputType}. Must be 'file' or 'frame'")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = self._model.process(frame_rgb)

        return results_face
