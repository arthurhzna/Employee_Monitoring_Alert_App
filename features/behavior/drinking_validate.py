from typing import Optional
import numpy as np
from inference.qwen3_vl_model import Qwen3VLMModel
from features.behavior.base_behavior import BaseBehavior


# class DrinkingValidate(BaseBehavior):

#     def get_prompt(self) -> str:
#         return "Is the person in this image drinking? Answer only 'yes' or 'no'."

# class DrinkingValidate(BaseBehavior):

#     def get_prompt(self) -> str:
#         return (
#             "Look carefully at the image. "
#             "Answer 'yes' ONLY if you clearly see a hand near the person's face "
#             "AND the hand is holding a drink container such as a glass, cup, bottle, can, or tumbler. "
#             "If you see a hand near the face but it is NOT holding any drink object, answer 'no'. "
#             "Respond with only: yes or no."
#         )

class DrinkingValidate(BaseBehavior):

    def get_prompt(self) -> str:
        return (
            "Look carefully at the image. "
            "Answer 'yes' ONLY if you clearly see a hand near the person's face "
            "AND the hand is holding a DRINK container such as a glass, cup, bottle, can, mug, or tumbler. "
            "If the hand is holding a spoon, fork, chopsticks, or any eating utensil (makanan), answer 'no'. "
            "If you are not clearly sure that it is a drink container, answer 'no'. "
            "Respond with only: yes or no."
        )