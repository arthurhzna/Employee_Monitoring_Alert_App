from dataclasses import dataclass
import numpy as np
from events.base_event import BaseEvent
from datetime import datetime
import core.config as config
import json

@dataclass
class HandInFaceEvent(BaseEvent):
    uuid: str
    frame: np.ndarray

    def __post_init__(self):
        self.path = ""

    def to_json(self) -> str:
        return json.dumps({
            "device_id": config.Config.device_id,
            "uuid": self.uuid,
            "path": self.path,
        }, default=str)