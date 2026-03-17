from typing import Any, Tuple
import numpy as np
import cv2


class HandInFaceFeature:
    def __init__(self, expand_ratio: float = 0.2) -> None:
        self.expand_ratio = expand_ratio

    def process(
        self,
        hand_results: Any,
        face_bbox: Tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> bool:
        if hand_results is None or not getattr(hand_results, "multi_hand_landmarks", None):
            return False

        ih, iw, _ = frame.shape
        fx1, fy1, fx2, fy2 = face_bbox

        fw = fx2 - fx1
        fh = fy2 - fy1

        expand_w = int(fw * self.expand_ratio)
        expand_h = int(fh * self.expand_ratio)

        fx1 = max(0, fx1 - expand_w)
        fy1 = max(0, fy1 - expand_h)
        fx2 = min(iw - 1, fx2 + expand_w)
        fy2 = min(ih - 1, fy2 + expand_h)

        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                lx = int(lm.x * iw)
                ly = int(lm.y * ih)

                if fx1 <= lx <= fx2 and fy1 <= ly <= fy2:
                    return True

        return False