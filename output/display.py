import cv2
import numpy as np

class Display:

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow("Display", frame)
        cv2.waitKey(1)