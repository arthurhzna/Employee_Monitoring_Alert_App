from typing import Dict
from datetime import datetime
from core.model import PersonData

class FaceDwellTimeFeature:
    
    def process(
        self,
        person: PersonData,
        attention: str,
        timestamp: datetime,
    ) -> None:

        if person.dwelltime_looking is None:
            person.dwelltime_looking = 0
        if person.dwelltime_not_looking is None:
            person.dwelltime_not_looking = 0
        if person.last_write_times_looking is None:
            person.last_write_times_looking = None
        if person.last_write_times_not_looking is None:
            person.last_write_times_not_looking = None

        if attention == "Looking":
            last_write = person.last_write_times_looking
            if last_write is None or (timestamp - last_write).total_seconds() >= 1:
                person.dwelltime_looking += 1  
                person.last_write_times_looking = timestamp  

        elif attention == "Not Looking":
            last_write = person.last_write_times_not_looking
            if last_write is None or (timestamp - last_write).total_seconds() >= 1:
                person.dwelltime_not_looking += 1  
                person.last_write_times_not_looking = timestamp  