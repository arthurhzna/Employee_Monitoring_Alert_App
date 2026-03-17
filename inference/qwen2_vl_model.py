import torch
from typing import Any, Optional, Union
from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
from transformers.image_utils import load_image
from inference.base_model import BaseModel
import numpy as np
import cv2
from PIL import Image


class Qwen2VLMModel(BaseModel):
    def __init__(self, model_name: str = "Qwen2-VL-2B-Instruct", device: str = "cpu") -> None:
        super().__init__(model_name, device)
        self._processor: Optional[Any] = None

    def load(self) -> None:
        if self._device == "cuda":
            torch.cuda.empty_cache()
        self._processor = AutoProcessor.from_pretrained(
            f"Qwen/{self._model_name}",
            cache_dir=f"./models/huggingface/{self._model_name}"
        )
        if self._device == "cuda":
            print("Loading model to CPU first (float16)...")
            self._model = AutoModelForImageTextToText.from_pretrained(
                f"Qwen/{self._model_name}",
                cache_dir=f"./models/huggingface/{self._model_name}",
                torch_dtype=torch.float16  
            )
            self._model = self._model.to("cpu")
            torch.cuda.empty_cache()
            print("Moving model to GPU...")
            self._model = self._model.to("cuda")
            print(f"✅ Model loaded on GPU: {next(self._model.parameters()).device}")
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                f"Qwen/{self._model_name}",
                cache_dir=f"./models/huggingface/{self._model_name}"
            )
            print(f"✅ Model loaded on CPU")
        
        self._is_loaded = True

    def predict(self, image: Union[str, np.ndarray],
        prompt: str,
        system_prompt: Optional[str] = None,
        inputType: str = "file") -> str:
        """
        Predict with support for:
        - inputType="file": image must be str (file path)
        - inputType="frame": image must be np.ndarray (crop frame)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        if inputType == "file":
            if not isinstance(image, str):
                raise TypeError(f"Expected str when inputType='file', got {type(image)}")
            image = load_image(image)
        elif inputType == "frame":
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected np.ndarray when inputType='frame', got {type(image)}")
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        else:
            raise ValueError(f"Invalid inputType: {inputType}. Must be 'file' or 'frame'")
        
        image = image.resize((384, 384))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        generation_config = GenerationConfig(
            max_new_tokens=200,  
            do_sample=False,  
            temperature=None,  
            top_p=None,  
            top_k=None  
        )
        outputs = self._model.generate(
            **inputs,
            generation_config=generation_config
        )
        result = self._processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True  
        )
        
        result = result.strip()
        result = result.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        return result