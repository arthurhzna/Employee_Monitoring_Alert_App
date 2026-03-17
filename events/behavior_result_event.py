from dataclasses import dataclass
from events.base_event import BaseEvent
import json

@dataclass
class BehaviorResultEvent(BaseEvent):
    uuid: str
    predict: str

    def to_json(self) -> str:
        return json.dumps({
            "uuid": self.uuid,
            "predict": self.predict,
        }, default=str)
