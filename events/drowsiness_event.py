from dataclasses import dataclass
from typing import Any, Dict
import time
import numpy as np
from events.base_event import BaseEvent
from datetime import datetime

@dataclass
class DrowsinessEvent(BaseEvent):
    track_id: int
    uuid: str
    timestamp: datetime
    frame: np.ndarray