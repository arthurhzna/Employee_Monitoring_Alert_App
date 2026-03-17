from dataclasses import dataclass
from events.base_event import BaseEvent
import json
import numpy as np
import core.config as config

@dataclass
class FaceRecogEvent(BaseEvent):
    uuid: str
    frame: np.ndarray

    def __post_init__(self):
        self.path = ""

    def to_json(self) -> str:
        return json.dumps({
            "device_id": config.Config.device_id,
            "uuid": self.uuid,
            "image": self.path,
        }, default=str)
