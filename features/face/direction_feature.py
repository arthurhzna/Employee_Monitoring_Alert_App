# domain logic - Face direction detection for dwell time
from typing import Optional, Any, Dict, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class FaceDirectionResult:
    attention: str       # "Looking" or "Not Looking"
    face_direction: str  # "Left", "Right", or "Center"

class FaceDirectionFeature:
    def __init__(
        self,
        threshold: float = 0.002
    ) -> None:
        self.threshold = threshold
    
    def process(
        self,
        faces: List,
        face_bbox_height: float
    ) -> FaceDirectionResult:

        normalized_distance_61_172, normalized_distance_291_397 = self._calculate_normalized_distances(
            faces, 
            face_bbox_height
        )
        
        return self.process_face_direction(
            normalized_distance_61_172,
            normalized_distance_291_397
        )
    
    def _calculate_normalized_distances(
        self,
        faces,
        face_bbox_height: float
    ) -> Tuple[Optional[float], Optional[float]]:

        normalized_distance_61_172 = None
        normalized_distance_291_397 = None

        if not faces or not faces.multi_face_landmarks:
            return None, None

        face_landmarks = faces.multi_face_landmarks[0]

        landmarks = face_landmarks.landmark

        if len(landmarks) > 397:

            p61 = np.array([landmarks[61].x, landmarks[61].y])
            p172 = np.array([landmarks[172].x, landmarks[172].y])

            distance_61_172 = np.linalg.norm(p61 - p172)
            normalized_distance_61_172 = distance_61_172 / face_bbox_height

            p291 = np.array([landmarks[291].x, landmarks[291].y])
            p397 = np.array([landmarks[397].x, landmarks[397].y])

            distance_291_397 = np.linalg.norm(p291 - p397)
            normalized_distance_291_397 = distance_291_397 / face_bbox_height

        return normalized_distance_61_172, normalized_distance_291_397
    
    def process_face_direction(
        self,
        normalized_distance_61_172: Optional[float],
        normalized_distance_291_397: Optional[float]
    ) -> FaceDirectionResult:
        if normalized_distance_61_172 is not None and normalized_distance_291_397 is not None:
            if normalized_distance_61_172 > self.threshold or normalized_distance_291_397 > self.threshold:
                attention = "Not Looking"
                if normalized_distance_61_172 is not None and normalized_distance_61_172 > self.threshold:
                    face_direction = "Left"
                elif normalized_distance_291_397 is not None and normalized_distance_291_397 > self.threshold:
                    face_direction = "Right"
                else:
                    face_direction = "Center"
            else:
                attention = "Looking"
                face_direction = "Center"
        else:
            attention = "Not Looking"
            face_direction = "Center"
        
        return FaceDirectionResult(
            attention=attention,
            face_direction=face_direction
        )
