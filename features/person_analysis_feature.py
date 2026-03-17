from typing import Optional, Tuple, Any
import uuid
import numpy as np
from datetime import datetime
from inference.yolo_model import YOLOModel
from preprocessing.crop import crop_person_with_expand
from utils.tensor_utils import get_tensor_value
from features.face_analysis_feature import FaceAnalysisFeature
from core.model import PersonDict, PersonData
from features.behavior.eating_validate import EatingValidate
from features.behavior.drinking_validate import DrinkingValidate
from features.behavior.smoking_validate import SmokingValidate
from utils.image_utils import draw_person_overlay
from events.person_disappeared_event import PersonDisappearedEvent
from events.event_bus import EventBus
class PersonAnalysisFeature:

    def __init__(
        self,
        person_detect_model: YOLOModel,
        face_analysis_feature: FaceAnalysisFeature,
        eating_validate_feature: EatingValidate,
        drinking_validate_feature: DrinkingValidate,
        smoking_validate_feature: SmokingValidate,
        event_bus: EventBus,
    ) -> None:
        self.person_detect_model = person_detect_model
        self.face_analysis_feature = face_analysis_feature
        self.eating_validate_feature = eating_validate_feature
        self.drinking_validate_feature = drinking_validate_feature
        self.smoking_validate_feature = smoking_validate_feature
        self.event_bus = event_bus
        self.person: PersonDict = {} 
        self.active_track_ids: set[int] = set()
    def load_models(self) -> None:
        self.person_detect_model.load()
        self.face_analysis_feature.load_models()

    def process(self, frame: np.ndarray, timestamp: datetime) -> None:
        persons_detect_results = self.person_detect_model.predict(frame)

        previous_active_track_ids = self.active_track_ids.copy()
        self.active_track_ids.clear()

        if persons_detect_results:
            for person in persons_detect_results:
                persons_bbox = person.boxes 
                for person_bbox in persons_bbox:

                    track_id = get_tensor_value(tensor=person_bbox.id)
                    
                    if track_id is None:
                        continue

                    if track_id not in self.person:
                        self.person[track_id] = PersonData() 
                        self.person[track_id].uuid = uuid.uuid4().hex 
                        
                    self.active_track_ids.add(track_id)

                    if self.person[track_id].timestamp is None:
                        self.person[track_id].timestamp = timestamp

                    px1, py1, px2, py2 = person_bbox.xyxy[0]
                    px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2) 

                    person_cropped_frame, px1, py1, px2, py2 = crop_person_with_expand(frame=frame, x1=px1, y1=py1, x2=px2, y2=py2, expand_factor=0.2)

                    if person_cropped_frame is None:
                        continue

                    self.person[track_id].person_bbox = (px1, py1, px2, py2)

                    self.face_analysis_feature.process(person_cropped_frame=person_cropped_frame, track_id=track_id, person=self.person[track_id], timestamp=timestamp)

                    draw_person_overlay(frame=frame, person_cropped_frame=person_cropped_frame, track_id=track_id, person=self.person[track_id])


        disappeared_track_ids = previous_active_track_ids - self.active_track_ids
        if disappeared_track_ids:
            for disappeared_id in disappeared_track_ids:
                person_disappeared_event = PersonDisappearedEvent(event_type="PERSON_DISAPPEARED", track_id=disappeared_id, person=self.person[disappeared_id])
                self.event_bus.publish(person_disappeared_event)
                self.person.pop(disappeared_id)


    def stop(self) -> None:
        self.person_detect_model.release()
        self.qwen3_vlm_model.release()
        self.face_analysis_feature.stop()