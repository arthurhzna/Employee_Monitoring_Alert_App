from __future__ import annotations
from typing import Any
from repository.device.device import DeviceRepository
from repository.person.person import PersonRepository
from repository.person.bbox import BboxRepository
from repository.person.dwelltime import DwelltimeRepository
from repository.person.behavior import BehaviorRepository
from repository.person.drowsiness import DrowsinessRepository
from repository.person.face_recog import FaceRecogRepository

class Tx:
    def __init__(self, conn: any) -> None:
        self._conn = conn
    def get_device(self) -> DeviceRepository:
        return DeviceRepository(self._conn)   
    def get_person(self) -> PersonRepository:
        return PersonRepository(self._conn)   
    def get_bbox(self) -> BboxRepository:
        return BboxRepository(self._conn)   
    def get_dwelltime(self) -> DwelltimeRepository:
        return DwelltimeRepository(self._conn)   
    def get_behavior(self) -> BehaviorRepository:
        return BehaviorRepository(self._conn)   
    def get_drowsiness(self) -> DrowsinessRepository:
        return DrowsinessRepository(self._conn)   
    def get_face_recog(self) -> FaceRecogRepository:
        return FaceRecogRepository(self._conn)   