# infrastructure layer - Main orchestration logic
from typing import Dict, Any, Optional, List
import numpy as np
from input.base_camera import BaseCamera
from output.display import Display
from preprocessing.resize import resize_frame
from datetime import datetime
from utils.fps_counter import FPSCounter
from features.person_analysis_feature import PersonAnalysisFeature

class Pipeline:
    def __init__(
        self,
        camera: BaseCamera,
        person_analysis_feature: PersonAnalysisFeature,
        display: Optional[Display] = None,
    ) -> None:
        self.camera = camera
        self.person_analysis_feature = person_analysis_feature
        self.display = display
        self.fps_counter = FPSCounter()

    def load_models(self) -> None:
        self.person_analysis_feature.load_models()

    def process_frame(self, frame: np.ndarray, timestamp: datetime) -> None:
        self.person_analysis_feature.process(frame=frame, timestamp=timestamp)

    def run(self) -> None:
        self._is_running = True
        self.camera.start()

        self.load_models()
        
        try:
            while self._is_running:
                grabbed, frame = self.camera.read()  
                if not grabbed or frame is None:
                    print("No frame grabbed")
                    continue

                timestamp = datetime.now()

                frame = resize_frame(frame=frame, width=640, height=480)

                self.fps_counter.update(timestamp=timestamp)
                
                self.process_frame(frame=frame, timestamp=timestamp)

                # print(f"FPS: {self.fps_counter.get_fps()}")
                
                if self.display:
                    self.display.show(frame=frame)
        
        except KeyboardInterrupt:
            print("Pipeline stopped by user")
        finally:
            self.stop()
    
    def stop(self) -> None:
        self._is_running = False
        if self.camera:
            self.camera.release()
