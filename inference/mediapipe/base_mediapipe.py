import mediapipe as mp
from typing import Optional, Any
from inference.base_model import BaseModel

class BaseMediaPipeModel(BaseModel):
    
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        super().__init__(model_name, device)
    
    def release(self) -> None:
        if self._model is not None:
            self._model.close()
        super().release()