from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from datetime import datetime

@dataclass
class PersonData:
    uuid: Optional[str] = None
    timestamp: Optional[datetime] = None
    dwelltime_looking: int = 0
    dwelltime_not_looking: int = 0
    last_write_times_looking: Optional[datetime] = None
    last_write_times_not_looking: Optional[datetime] = None
    person_bbox: Optional[Tuple[int, int, int, int]] = None  
    face_bbox: Optional[Tuple[int, int, int, int]] = None  
    face_recog_trigger: bool = False
    face_recog: str = "Unknown"
    hand_in_face_previous_state: Optional[bool] = None
    drowsiness_previous_state: Optional[bool] = None
    eating_detected: bool = False
    drinking_detected: bool = False
    smoking_detected: bool = False
    
PersonDict = Dict[int, PersonData]