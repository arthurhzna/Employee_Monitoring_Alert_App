from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
from abc import ABC
import time

@dataclass
class BaseEvent(ABC):
    event_type: str

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                "event_type": self.event_type,
            }
        except Exception as e:
            return {"error": str(e)}

    def to_json(self) -> str:
        try:
            return json.dumps(self.to_dict(), default=str)
        except Exception as e:
            return json.dumps({"error": f"JSON serialization failed: {e}"})