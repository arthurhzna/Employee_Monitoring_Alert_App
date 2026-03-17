import numpy as np
from typing import Any

class DrowsinessFeature:    
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]
    EAR_THRESHOLD = 0.15
    
    @staticmethod
    def calculate_EAR(points: list) -> float:
        d1 = np.linalg.norm(points[1] - points[5])
        d2 = np.linalg.norm(points[2] - points[4])
        d3 = np.linalg.norm(points[0] - points[3])
        return (d1 + d2) / (2 * d3)
    
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

        ear_left = self.calculate_EAR([points[i] for i in self.LEFT_EYE])
        ear_right = self.calculate_EAR([points[i] for i in self.RIGHT_EYE])
        is_drowsy = ear_left < self.EAR_THRESHOLD or ear_right < self.EAR_THRESHOLD
        
        return is_drowsy