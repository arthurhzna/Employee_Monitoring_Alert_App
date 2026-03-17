from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import numpy as np
import threading


class BaseCamera(ABC):
    def __init__(self) -> None:
        self._is_running: bool = False
        self._grabbed: bool = False
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[Any] = None

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    def start(self) -> bool: 
        self._is_running = True
        return self._is_running

    def stop(self) -> bool:  
        self._is_running = False
        return self._is_running

    @abstractmethod
    def release(self) -> None:
        pass