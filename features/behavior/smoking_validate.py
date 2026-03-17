from typing import Optional
import numpy as np
from inference.qwen3_vl_model import Qwen3VLMModel
from features.behavior.base_behavior import BaseBehavior


# class SmokingValidate(BaseBehavior):
        
#     def get_prompt(self) -> str:
#         return "Is the person in this image smoking a cigarette? Answer only 'yes' or 'no'."

class SmokingValidate(BaseBehavior):
        
    def get_prompt(self) -> str:
        return (
            "Look carefully at the image. "
            "Answer 'yes' ONLY if you clearly see the person SMOKING a cigarette or similar object "
            "(for example: cigarette in the hand near the mouth, cigarette in the mouth, or smoke visible). "
            "If you only see a hand near the mouth but no clear cigarette or smoking object, answer 'no'. "
            "Respond with only: yes or no."
        )