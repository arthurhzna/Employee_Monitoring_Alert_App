from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from inference.qwen3_vl_model import Qwen3VLMModel

class BaseBehavior(ABC):
    
    @abstractmethod
    def get_prompt(self) -> str:
        pass
    
    def validate(self, image: np.ndarray, vlm_model: Qwen3VLMModel) -> bool:
        
        prompt = self.get_prompt()
        
        try:
            result = vlm_model.predict(
                image=image,
                prompt=prompt,
                inputType="frame"
            )
            
            result_lower = result.lower().strip()
            
            if "yes" in result_lower:
                return True
            elif "no" in result_lower:
                return False
            else:
                return False
                
        except Exception as e:
            print(f"Error in behavior validation: {e}")
            return False

