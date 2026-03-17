from core.config import init_config
from core.pipeline import Pipeline

from input.rtsp_camera import RTSPCamera
from output.display import Display

from inference.yolo_model import YOLOModel
from inference.mediapipe.face_mesh import FaceMeshMediapipeModel
from inference.mediapipe.hand_mesh import HandMeshMediapipeModel
from inference.mediapipe.face_detect import FaceDetectMediapipeModel 

from features.behavior.eating_validate import EatingValidate
from features.behavior.drinking_validate import DrinkingValidate
from features.behavior.smoking_validate import SmokingValidate

from features.person_analysis_feature import PersonAnalysisFeature
from features.face_analysis_feature import FaceAnalysisFeature
from features.face.direction_feature import FaceDirectionFeature
from features.face.dwelltime_feature import FaceDwellTimeFeature
from features.hand_in_face_feature import HandInFaceFeature
from features.face.drowsiness_feature import DrowsinessFeature
from features.face.yawning_feature import YawningFeature

from core.container import Container
import core.config as config

def main():
    init_config()
    container = Container()
    container.setup()

    person_detect_model = YOLOModel(
        model_name="yolo_person_detector",
        device="gpu",
        model_path="./models/yolo/universal/yolo11n/yolo11n.pt",
        classNames=[0],
        conf=0.5,
        track=True,
        track_type="bytetrack.yaml",
    )

    face_detect_model = FaceDetectMediapipeModel(device="gpu", min_detection_confidence=0.9)
    face_mesh_model = FaceMeshMediapipeModel(device="gpu")
    hand_mesh_model = HandMeshMediapipeModel(device="gpu")

    eating_validate_feature = EatingValidate()
    drinking_validate_feature = DrinkingValidate()
    smoking_validate_feature = SmokingValidate()

    face_direction_feature = FaceDirectionFeature()
    face_dwell_time_feature = FaceDwellTimeFeature()
    hand_to_face_validate_feature = HandInFaceFeature()

    drowsiness_feature = DrowsinessFeature()
    yawning_feature = YawningFeature()

    face_analysis_feature = FaceAnalysisFeature(
        face_detect_model=face_detect_model,
        face_mesh_model=face_mesh_model,
        hand_mesh_model=hand_mesh_model,
        
        face_direction_feature=face_direction_feature,
        face_dwell_time_feature=face_dwell_time_feature,
        hand_to_face_validate_feature=hand_to_face_validate_feature,
        drowsiness_feature=drowsiness_feature,
        yawning_feature=yawning_feature,

        event_bus=container.get("event_bus"),
        redis_result_consumer=container.get("redis_result_consumer"),
    )

    person_analysis_feature = PersonAnalysisFeature(
        person_detect_model=person_detect_model,

        face_analysis_feature=face_analysis_feature,
        eating_validate_feature=eating_validate_feature,
        drinking_validate_feature=drinking_validate_feature,
        smoking_validate_feature=smoking_validate_feature,

        event_bus=container.get("event_bus"),
    )
    print(f"config.Config.camera_url: {config.Config.camera_url}")
    camera = RTSPCamera(config.Config.camera_url)
    display = Display()

    pipeline = Pipeline(
        camera=camera,
        person_analysis_feature=person_analysis_feature,
        display=display,
    )
    pipeline.run()

if __name__ == "__main__":
    main()