# core/event_handlers.py
"""Event handlers untuk berbagai event types"""

from events.hand_in_face_event import HandInFaceEvent
from events.drowsiness_event import DrowsinessEvent
from events.person_disappeared_event import PersonDisappearedEvent
from events.face_recog_result_event import FaceRecogResultEvent
from events.behavior_result_event import BehaviorResultEvent
from externals.redis_client import RedisClient
from repository.data_store import DataStore
from repository.tx import Tx
from externals.tapo_cctv_speaker_client import TapoCctvSpeakerClient
from events.face_recog_event import FaceRecogEvent
from typing import Callable

from utils.image_utils import write_image
from core.worker import submit_task_to_worker

import threading
_audio_play_thread_lock = threading.Lock()

def hand_in_face_event_handler(
    redis_client: RedisClient
) -> Callable[[HandInFaceEvent], None]:
    def handle(event: HandInFaceEvent):
        print(f"[HAND_IN_FACE] Hand in face detected for uuid={event.uuid}")
        path = f"data/hand_in_face/{event.uuid}.jpg"
        event.path = path
        write_image(event.frame, path)
        redis_client.rpush("behavior_queue", event.to_json())
    return handle

def face_recog_event_handler(redis_client: RedisClient
) -> Callable[[FaceRecogEvent], None]:
    def handle(event: FaceRecogEvent):
        print(f"[FACE_RECOG] Face recog detected for uuid={event.uuid}")
        path = f"data/face_recog/{event.uuid}.jpg"
        event.image = path
        write_image(event.frame, path)
        redis_client.rpush("face_recog_queue", event.to_json())
    return handle

def drowsiness_event_handler(
    tapo_cctv_speaker_client: TapoCctvSpeakerClient
)-> Callable[[DrowsinessEvent], None]:
    def handle(event: DrowsinessEvent):
        print(f"[DROWSINESS] Drowsiness detected for track_id={event.track_id}")
        if not _audio_play_thread_lock.acquire(blocking=False):
            return
        def _play_audio():
            try:
                tapo_cctv_speaker_client.streamovat("./audio/fokus.wav")
            finally:  
                _audio_play_thread_lock.release()
        submit_task_to_worker(_play_audio) 
  
    return handle

def person_disappeared_event_handler(data_store: DataStore, device_id_db: int)-> Callable[[PersonDisappearedEvent], None]:
    def handle(event: PersonDisappearedEvent):
        print(f"[PERSON_DISAPPEARED] Person {event.track_id} disappeared")
        def save(tx: Tx):
            person_id = tx.get_person().insert_person(person=event.person, device_id=device_id_db)
            if person_id is not None:
                tx.get_bbox().insert_bbox(
                    person_id=person_id,
                    width=event.person.person_bbox[2] - event.person.person_bbox[0],   
                    height=event.person.person_bbox[3] - event.person.person_bbox[1]   
                )
                tx.get_dwelltime().insert_dwelltime(
                    person_id=person_id,
                    dwelling_looking=event.person.dwelltime_looking,
                    dwelling_not_looking=event.person.dwelltime_not_looking
                )

        submit_task_to_worker(data_store.atomic, save)

    return handle

def face_recog_result_event_handler(data_store: DataStore) -> Callable[[FaceRecogResultEvent], None]:
    def handle(event: FaceRecogResultEvent):
        print(f"[FACE_RECOG_RESULT] Face recog result detected for uuid={event.uuid}")
        def save(tx: Tx):
            person_id = tx.get_person().get_person_id(uuid=event.uuid)
            if person_id is not None:
                tx.get_face_recog().insert_face_recog(person_id=person_id, predict=event.predict)

        submit_task_to_worker(data_store.atomic, save) 

    return handle

def behavior_result_event_handler(data_store: DataStore) -> Callable[[BehaviorResultEvent], None]:
    def handle(event: BehaviorResultEvent):
        def save(tx: Tx):
            person_id = tx.get_person().get_person_id(uuid=event.uuid)
            if person_id is not None:
                tx.get_behavior().insert_behavior(person_id=person_id, predict=event.predict)

        submit_task_to_worker(data_store.atomic, save) 

    return handle