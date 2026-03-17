import time
from typing import Optional
import numpy as np
import cv2
from datetime import datetime

class FPSCounter:
    def __init__(self) -> None:
        self.prev_time: Optional[float] = None
        self.current_fps: float = 0.0

    def update(self, timestamp: datetime) -> float:
        if self.prev_time is not None:
            frame_time = (timestamp - self.prev_time).total_seconds()
            self.current_fps = 1.0 / frame_time if frame_time > 0.0 else 0.0


        self.prev_time = timestamp
        return self.current_fps

    def get_fps(self) -> float:
        return self.current_fps

    def reset(self) -> None:
        self.prev_time = None
        self.current_fps = 0.0