import torch
from inference.base_model import BaseModel
from ultralytics import YOLO
import numpy as np
from typing import Dict, Any, List, Union
import cv2
import gc

class YOLOModel(BaseModel):
    """
    trackType with support for:
    - trackType="bytetrack.yaml": ByteTrack tracking algorithm
    - trackType="botsort.yaml": BoTSORT tracking algorithm
    """
    def __init__(self, model_name: str,
        device: str = "cpu",
        model_path: str = "./models/yolo/head/yolo11n/head.pt",
        classNames: List[str] = [],
        conf: float = 0.9,
        track: bool = False,
        track_type: str = "bytetrack.yaml",
        ) -> None:
        super().__init__(model_name, device)
        self._model_path: str = model_path
        self._classNames: List[str] = classNames
        self._conf: float = conf
        self._track: bool = track
        self._track_type: str = track_type

    def load(self) -> None:
        if self._classNames == []:
            raise ValueError("classNames is empty")

        self._model = YOLO(self._model_path)
        self._is_loaded = True

        if self._device == "cuda":
            self._model.to("cuda")

    def predict(self, image: Union[str, np.ndarray],
        inputType: str = "frame"
        ) -> Dict[int, Dict[str, Any]]:
        """
        Predict with support for:
        - inputType="file": frame must be str (file path)
        - inputType="frame": frame must be np.ndarray (crop frame)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if inputType == "file":
            if not isinstance(image, str):
                raise TypeError(f"Expected str when inputType='file', got {type(image)}")
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from path: {image}")
        elif inputType == "frame":
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected np.ndarray when inputType='frame', got {type(image)}")
            image = image.copy() 
        else:
            raise ValueError(f"Invalid inputType: {inputType}. Must be 'file' or 'frame'")
        
        if self._track:
            results = self._model.track(image, persist=True, classes=self._classNames, conf=self._conf, tracker=self._track_type)
        else:
            results = self._model.predict(image, persist=True, classes=self._classNames, conf=self._conf)
        
        return results

    def release(self) -> None:
        del self._model

        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
