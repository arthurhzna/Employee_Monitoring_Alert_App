from __future__ import annotations
from datetime import datetime


class DrowsinessRepository:
    def __init__(self, conn: any) -> None:
        self._conn = conn

    def insert_drowsiness(self, person_id: int, timestamp: datetime) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO drowsiness (timestamp, person_id)
                VALUES (%s, %s)
                """,
                (timestamp, person_id),
            )
        except Exception:
            raise
        finally:
            cur.close()