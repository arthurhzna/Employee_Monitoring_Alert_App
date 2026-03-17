import numpy as np
from typing import Any

class YawningFeature:
    MOUTH = [61, 39, 0, 269, 291, 405, 17, 181]
    MAR_THRESHOLD = 1.2
    
    @staticmethod
    def calculate_MAR(points: list) -> float:
        d1 = np.linalg.norm(points[1] - points[7])
        d2 = np.linalg.norm(points[2] - points[6])
        d3 = np.linalg.norm(points[3] - points[5])
        d4 = np.linalg.norm(points[0] - points[4])
        return (d1 + d2 + d3) / (2 * d4)
    
    def process(self, landmarks: Any) -> bool:
        if (
            landmarks is None
            or not hasattr(landmarks, "multi_face_landmarks")
            or not landmarks.multi_face_landmarks
        ):
            return False

        face_landmarks = landmarks.multi_face_landmarks[0].landmark
        if len(face_landmarks) < 400:
            return False

        points = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks])

        mar = self.calculate_MAR([points[i] for i in self.MOUTH])
        is_yawning = mar > self.MAR_THRESHOLD
        
        return is_yawning