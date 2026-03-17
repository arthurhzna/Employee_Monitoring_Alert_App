import numpy as np
from datetime import datetime

from inference.yolo_model import YOLOModel
from inference.mediapipe.face_mesh import FaceMeshMediapipeModel
from inference.mediapipe.hand_mesh import HandMeshMediapipeModel
from inference.mediapipe.face_detect import FaceDetectMediapipeModel

from features.face.direction_feature import FaceDirectionFeature
from features.face.dwelltime_feature import FaceDwellTimeFeature
from features.hand_in_face_feature import HandInFaceFeature
from features.face.drowsiness_feature import DrowsinessFeature
from features.face.yawning_feature import YawningFeature

from core.model import PersonData
from preprocessing.crop import crop_face

from events.event_bus import EventBus
from events.hand_in_face_event import HandInFaceEvent
from events.drowsiness_event import DrowsinessEvent
from events.face_recog_event import FaceRecogEvent
from events.face_recog_result_event import FaceRecogResultEvent
from events.behavior_result_event import BehaviorResultEvent

import core.config as config
from externals.redis_handlers.redis_result_consumer import RedisResultConsumer

class FaceAnalysisFeature:
    def __init__(
        self,
        face_detect_model: FaceDetectMediapipeModel,
        face_mesh_model: FaceMeshMediapipeModel,
        hand_mesh_model: HandMeshMediapipeModel,
        face_direction_feature: FaceDirectionFeature,
        face_dwell_time_feature: FaceDwellTimeFeature,
        hand_to_face_validate_feature: HandInFaceFeature,
        drowsiness_feature: DrowsinessFeature,
        yawning_feature: YawningFeature,
        event_bus: EventBus,
        redis_result_consumer: RedisResultConsumer,
    ) -> None:
        self.face_detect_model = face_detect_model
        self.face_mesh_model = face_mesh_model
        self.hand_mesh_model = hand_mesh_model
        self.face_direction_feature = face_direction_feature
        self.face_dwell_time_feature = face_dwell_time_feature
        self.hand_to_face_validate_feature = hand_to_face_validate_feature
        self.drowsiness_feature = drowsiness_feature
        self.yawning_feature = yawning_feature
        self.event_bus = event_bus
        self.redis_result_consumer = redis_result_consumer
    def load_models(self) -> None:
        self.face_detect_model.load()
        self.face_mesh_model.load()
        self.hand_mesh_model.load()

    def process(self, person_cropped_frame: np.ndarray, track_id: int, person: PersonData, timestamp: datetime) -> None:
        face_recog_result = self.redis_result_consumer.consume_results_from_redis_lpop(f"face_recog_results_{config.Config.device_id}")
        if face_recog_result is not None:
            face_recog_result_event = FaceRecogResultEvent(
                event_type="FACE_RECOG_RESULT_DETECTED",
                uuid=face_recog_result["uuid"],
                predict=face_recog_result["predict"],
            )
            self.event_bus.publish(face_recog_result_event)

            if face_recog_result["uuid"] == person.uuid: 
                person.face_recog = face_recog_result["predict"]

        behavior_result = self.redis_result_consumer.consume_results_from_redis_lpop(f"behavior_results_{config.Config.device_id}")
        if behavior_result is not None:
            behavior_result_event = BehaviorResultEvent(
                event_type="BEHAVIOR_RESULT_DETECTED",
                uuid=behavior_result["uuid"],
                predict=behavior_result["predict"],
            )
            self.event_bus.publish(behavior_result_event)
            if behavior_result["uuid"] == person.uuid: 
                if behavior_result["predict"] == "eating":  
                    person.eating_detected = True
                elif behavior_result["predict"] == "drinking":
                    person.drinking_detected = True
                elif behavior_result["predict"] == "smoking":
                    person.smoking_detected = True
                else: 
                    person.eating_detected = False
                    person.drinking_detected = False
                    person.smoking_detected = False

        faces_detect_results = self.face_detect_model.predict(frame=person_cropped_frame, inputType="frame")

        face_attention = "Not Looking"
        hand_to_face_result = False
        drowsiness_result = False
        yawning_result = False

        if faces_detect_results.detections:
            for face in faces_detect_results.detections:
                face_bbox = face.location_data.relative_bounding_box

                ph, pw, _ = person_cropped_frame.shape
                fx1, fy1, fx2, fy2 = int(face_bbox.xmin * pw), int(face_bbox.ymin * ph), int((face_bbox.xmin + face_bbox.width) * pw), int((face_bbox.ymin + face_bbox.height) * ph)

                person.face_bbox = (fx1, fy1, fx2, fy2)
        
                face_cropped_frame = crop_face(frame=person_cropped_frame, x1=fx1, y1=fy1, x2=fx2, y2=fy2)

                if face_cropped_frame is None:
                    continue

                face_mesh_results = self.face_mesh_model.predict(frame=face_cropped_frame, inputType="frame")

                hand_mesh_results = self.hand_mesh_model.predict(frame=person_cropped_frame, inputType="frame")

                hand_to_face_result = self.hand_to_face_validate_feature.process(hand_results=hand_mesh_results, face_bbox=(fx1, fy1, fx2, fy2), frame=person_cropped_frame)

                face_direction_result = self.face_direction_feature.process(faces=face_mesh_results, face_bbox_height=fy2 - fy1)

                drowsiness_result = self.drowsiness_feature.process(landmarks=face_mesh_results)
                yawning_result = self.yawning_feature.process(landmarks=face_mesh_results)

                if face_direction_result is None:
                    continue

                face_attention = face_direction_result.attention

                if face_attention == "Looking":
                    if not person.face_recog_trigger:
                        person.face_recog_trigger = True

                        face_recog_event = FaceRecogEvent(
                            event_type="FACE_RECOG_DETECTED",
                            uuid=person.uuid,
                            frame=face_cropped_frame,
                        )
                        self.event_bus.publish(face_recog_event)

        self.face_dwell_time_feature.process(person=person, attention=face_attention, timestamp=timestamp)

        if not person.hand_in_face_previous_state and hand_to_face_result:
            hand_in_face_event = HandInFaceEvent(
                event_type="HAND_IN_FACE_DETECTED",
                uuid=person.uuid,
                frame=person_cropped_frame,
            )
            self.event_bus.publish(hand_in_face_event)

        person.hand_in_face_previous_state = hand_to_face_result

        if not person.drowsiness_previous_state and drowsiness_result and yawning_result:
            drowsiness_event = DrowsinessEvent(
                event_type="DROWSINESS_DETECTED",
                track_id=track_id,
                uuid=person.uuid,
                timestamp=person.timestamp,
                frame=face_cropped_frame,
            )
            self.event_bus.publish(drowsiness_event)

        person.drowsiness_previous_state = drowsiness_result and yawning_result

    def stop(self) -> None:
        self.face_detect_model.release()
        self.face_mesh_model.release()
        self.hand_mesh_model.release()
