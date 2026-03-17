# core/state_manager.py
from typing import List, Dict, Any, Optional
from threading import Lock
from dataclasses import dataclass, field


@dataclass
class DeviceState:
    is_registered: bool = False
    device_id_db: int = None

@dataclass
class ScreenshotState:
    url: Optional[str] = None
    flag: bool = False

class StateManager:
    def __init__(self):
        self._lock : Lock = Lock()
        self._device_state : DeviceState = DeviceState()
        self._screenshot_state : ScreenshotState = ScreenshotState()
    
    def update_device_registration(self, is_registered: bool) -> None:
        with self._lock:
            self._device_state.is_registered = is_registered
    
    def get_device_registration_state(self) -> DeviceState:
        with self._lock:
            return self._device_state.is_registered
    
    def get_screenshot_state(self) -> ScreenshotState:
        with self._lock:
            return ScreenshotState(
                url=self._screenshot_state.url,
                flag=self._screenshot_state.flag
            )

    def update_device_id_db(self, device_id_db: int) -> None:
        with self._lock:
            self._device_state.device_id_db = device_id_db

    def get_device_id_db(self) -> Optional[int]:
        with self._lock:
            return self._device_state.device_id_db