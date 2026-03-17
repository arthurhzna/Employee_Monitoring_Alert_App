from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseModel(ABC):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name: str = model_name
        if device:
            if device not in ["cpu", "gpu", "cuda"]:
                raise ValueError("Invalid device")
            if device == "gpu":
                self._device: str = "cuda"
            else:
                self._device: str = "cpu"
        self._model: Optional[Any] = None
        self._is_loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        pass

    def release(self) -> None:
        self._model = None
        self._is_loaded = False