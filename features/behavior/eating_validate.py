from typing import Optional
import numpy as np
from inference.qwen3_vl_model import Qwen3VLMModel
from features.behavior.base_behavior import BaseBehavior


# class EatingValidate(BaseBehavior):
    
#     def get_prompt(self) -> str:
#         return "Is the person in this image eating food? Answer only 'yes' or 'no'."

class EatingValidate(BaseBehavior):
    
    def get_prompt(self) -> str:
        return (
            "Look carefully at the image. "
            "Answer 'yes' ONLY if you clearly see the person eating FOOD "
            "(for example: food visible near the mouth, hand holding food, "
            "or food very close to the mouth). "
            "If you only see a hand near the mouth but no clear food object, answer 'no'. "
            "Respond with only: yes or no."
        )