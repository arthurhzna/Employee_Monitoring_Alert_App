from dataclasses import dataclass, asdict
from typing import Dict, Any
from events.base_event import BaseEvent
from core.model import PersonData
import time
import json

@dataclass
class PersonDisappearedEvent(BaseEvent):
    track_id: int
    person: PersonData

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self._person)

    def to_json(self) -> str:
        return json.dumps({
            "uuid": self.person.uuid,
            "track_id": self.track_id,
            "timestamp": self.person.timestamp.isoformat(),
            "person_bbox": self.person.person_bbox,
            "face_bbox": self.person.face_bbox,
            "face_recog": self.person.face_recog,
        }, default=str)
