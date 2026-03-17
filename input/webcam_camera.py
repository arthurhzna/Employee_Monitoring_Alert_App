import cv2
import numpy as np
from typing import Optional
from input.base_camera import BaseCamera
import threading


class WebcamCamera(BaseCamera):
    def __init__(self, device: int = 0) -> None:
        super().__init__()
        self._cap = cv2.VideoCapture(device)

    def start(self) -> "WebcamCamera":
        super().start()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def _update(self) -> None:
        while self._is_running:
            grabbed, frame = self._cap.read()
            if not grabbed:
                continue
            with self._lock:
                self._grabbed = grabbed
                self._frame = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return self._grabbed, self._frame.copy() if self._frame is not None else None

    def release(self) -> None:
        super().stop()
        if self._thread is not None:
            self._thread.join()
        if self._cap is not None:
            self._cap.release()