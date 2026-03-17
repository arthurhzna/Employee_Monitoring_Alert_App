from __future__ import annotations
from typing import Optional
from core.model import PersonData


class PersonRepository:
    def __init__(self, conn: any) -> None:
        self._conn = conn

    def insert_person(self, person: PersonData, device_id: int) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO person (uuid, timestamp, device_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (uuid) DO NOTHING
                RETURNING person_id
                """,
                (person.uuid, person.timestamp, device_id),
            )
            row = cur.fetchone()
            return row[0] if row else self.get_person_id(person.uuid)
        except Exception:
            raise
        finally:
            cur.close()

    def get_person_id(self, uuid: str) -> Optional[int]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT person_id FROM person WHERE uuid = %s",
                (uuid,),
            )
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            cur.close()

    def get_person(self, uuid: str) -> Optional[tuple]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT * FROM person WHERE uuid = %s",
                (uuid,),
            )
            return cur.fetchone()
        finally:
            cur.close()