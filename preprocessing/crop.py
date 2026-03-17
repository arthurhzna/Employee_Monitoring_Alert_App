import numpy as np
from typing import Tuple, Optional
from core.model import PersonData

def crop_head_with_expand(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    expand_factor: float = 0.2,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]

    width = x2 - x1
    height = y2 - y1

    x1 = int(x1 - width * expand_factor)
    y1 = int(y1 - height * expand_factor)
    x2 = int(x2 + width * expand_factor)
    y2 = int(y2 + height * expand_factor)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], x1, y1, x2, y2

def crop_person_with_expand(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    expand_factor: float = 0.2,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    h, w = frame.shape[:2]

    width = x2 - x1
    height = y2 - y1

    x1 = int(x1 - width * expand_factor)
    y1 = int(y1 - height * expand_factor)
    x2 = int(x2 + width * expand_factor)
    y2 = int(y2 + height * expand_factor)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    return frame[y1:y2, x1:x2], x1, y1, x2, y2

def crop_face(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray:
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    face_cropped = frame[y1:y2, x1:x2]
    if face_cropped.size > 0:
        return face_cropped

    return None