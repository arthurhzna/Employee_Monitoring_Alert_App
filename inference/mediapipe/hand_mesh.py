import cv2
import mediapipe as mp
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union
from inference.mediapipe.base_mediapipe import BaseMediaPipeModel


class HandMeshMediapipeModel(BaseMediaPipeModel):

    def __init__(self, model_name: str = "mediapipe_hand_mesh", device: str = "cpu") -> None:
        super().__init__(model_name, device)

    def load(self) -> None:
        self._model = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._is_loaded = True

    def predict(self, frame: Union[str, np.ndarray], inputType: str = "frame") -> Dict[str, Any]:
        """
        Predict with support for:
        - inputType="file": frame must be str (file path)
        - inputType="frame": frame must be np.ndarray (crop frame)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if inputType == "file":
            if not isinstance(frame, str):
                raise TypeError(f"Expected str when inputType='file', got {type(frame)}")
            image = cv2.imread(frame)
            if image is None:
                raise ValueError(f"Could not load image from path: {frame}")
        elif inputType == "frame":
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"Expected np.ndarray when inputType='frame', got {type(frame)}")
            image = frame
        else:
            raise ValueError(f"Invalid inputType: {inputType}. Must be 'file' or 'frame'")

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_results = self._model.process(frame_rgb)

        return hands_results