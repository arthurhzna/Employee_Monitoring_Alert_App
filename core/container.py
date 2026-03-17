from typing import Dict, Any
from events.event_bus import EventBus
from core.state_manager import StateManager
from core.database import init_database
from repository.data_store import DataStore
from repository.tx import Tx

# MQTT imports
# from externals.mqtt_client import MQTTClient
# from externals.mqtt_handlers.device_registration_handler import DeviceRegistrationHandler

from externals.tapo_cctv_speaker_client import TapoCctvSpeakerClient
from externals.redis_client import RedisClient
from externals.redis_handlers.redis_result_consumer import RedisResultConsumer

import core.config as config

# Event handlers
from core.event_handlers import (
    hand_in_face_event_handler, 
    drowsiness_event_handler,
    face_recog_event_handler,
    person_disappeared_event_handler,
    face_recog_result_event_handler,
    behavior_result_event_handler,
)


class Container:
    def __init__(self):
        self._services: Dict[str, Any] = {}

    def register(self, name: str, instance: Any):
        self._services[name] = instance

    def get(self, name: str) -> Any:
        if name in self._services:
            return self._services[name]
        raise ValueError(f"Service {name} not found")
    
    def setup(self):

        # # Database
        pool = init_database() 
        self._services["pool"] = pool

        # # Repository db
        data_store = DataStore(pool=pool)
        self._services["repository_registry"] = data_store

        # State Manager
        state_manager = StateManager()
        self._services["state_manager"] = state_manager

        def fetch_device_info(tx: Tx):
            device_id_row = tx.get_device().get_device_id_db(device_name=config.Config.device_id)
            
            if device_id_row is None:
                tx.get_device().insert_device(device_name=config.Config.device_id)
                device_id_row = tx.get_device().get_device_id_db(device_name=config.Config.device_id)

            is_registered_row = tx.get_device().get_is_registered(device_name=config.Config.device_id)

            return (
                device_id_row[0] if device_id_row else None,
                is_registered_row[0] if is_registered_row else False,
            )

        device_id_db, is_registered = data_store.atomic(fetch_device_info)

        state_manager.update_device_id_db(device_id_db=device_id_db)
        state_manager.update_device_registration(is_registered=is_registered)

        # Event Bus
        event_bus = EventBus()
        self._services["event_bus"] = event_bus

        # Redis Client
        redis_client = RedisClient()
        redis_client.connect()
        self._services["redis_client"] = redis_client

        redis_consumer = RedisResultConsumer(redis_client=redis_client)
        self._services["redis_result_consumer"] = redis_consumer

        # MQTT Client
        # mqtt_client = MQTTClient()
        
        # if not mqtt_client.connect():
        #     print("Warning: MQTT connection failed, continuing without MQTT...")
        
        # mqtt_client.register_handler(
        #     DeviceRegistrationHandler(state_manager=state_manager)
        # )

        # mqtt_client.subscribe(f"alert/subscribe/{Config.device_id}")
        # mqtt_client.subscribe(f"alert/{Config.device_id}/screenshoot")

        # mqtt_client.register_handler(
        #     DeviceRegistrationHandler(state_manager=state_manager)
        # )

        # self._services["mqtt_client"] = mqtt_client

        # Speaker Tapo CCTV Client
        tapo_cctv_speaker_client = TapoCctvSpeakerClient()
        self._services["tapo_cctv_speaker_client"] = tapo_cctv_speaker_client
        
        event_bus.subscribe("HAND_IN_FACE_DETECTED", hand_in_face_event_handler(redis_client=redis_client))
        event_bus.subscribe("DROWSINESS_DETECTED", drowsiness_event_handler(tapo_cctv_speaker_client=tapo_cctv_speaker_client))
        event_bus.subscribe("FACE_RECOG_DETECTED", face_recog_event_handler(redis_client=redis_client))
        event_bus.subscribe("PERSON_DISAPPEARED", person_disappeared_event_handler(data_store=data_store, device_id_db=state_manager.get_device_id_db()))
        event_bus.subscribe("FACE_RECOG_RESULT_DETECTED", face_recog_result_event_handler(data_store=data_store))
        event_bus.subscribe("BEHAVIOR_RESULT_DETECTED", behavior_result_event_handler(data_store=data_store))