import torch
from typing import Dict, Any, Optional, Union, List
from transformers import ViTImageProcessor, ViTForImageClassification
from inference.base_model import BaseModel
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
#list model name
#nateraw/vit-age-classifier
#rizvandwiki/gender-classification

@dataclass
class ViTImageClassificationResult:
    label: str
    confidence: float

class ViTTransformerModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu"
    ) -> None:
        super().__init__(model_name, device)
        self._cache_dir = f"./models/huggingface/{model_name}"
        self._processor: Optional[Any] = None

    def load(self) -> None:
        if self._device == "cuda":
            torch.cuda.empty_cache()
        
        self._processor = ViTImageProcessor.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir
        )
        if self._device == "cuda":
            self._model = ViTForImageClassification.from_pretrained(
                self._model_name,
                cache_dir=self._cache_dir
            )
            self._model.to("cuda")
        else:
            self._model = ViTForImageClassification.from_pretrained(
                self._model_name,
                cache_dir=self._cache_dir
            )
            self._model.to("cpu")

        self._model.eval()
        self._is_loaded = True

    def predict(self, image: Union[str, np.ndarray],
        inputType: str = "frame"
        ) -> Dict[int, Dict[str, Any]]:

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
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            raise ValueError(f"Invalid inputType: {inputType}. Must be 'file' or 'frame'")

        inputs = self._processor(
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self._device_obj) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

        label = self._model.config.id2label[pred_idx]

        return ViTImageClassificationResult(
            label=label,
            confidence=round(confidence, 4)
        )