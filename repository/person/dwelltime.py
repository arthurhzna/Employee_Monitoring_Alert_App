from __future__ import annotations
from typing import Optional


class DwelltimeRepository:
    def __init__(self, conn: any) -> None:
        self._conn = conn

    def insert_dwelltime(
        self,
        person_id: int,
        dwelling_looking: int,
        dwelling_not_looking: int,
    ) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO dwelling_time (dwelling_looking, dwelling_not_looking, person_id)
                VALUES (%s, %s, %s)
                """,
                (dwelling_looking, dwelling_not_looking, person_id),
            )
        except Exception:
            raise
        finally:
            cur.close()

    def get_dwelltime(self, person_id: int) -> Optional[tuple]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT * FROM dwelling_time WHERE person_id = %s",
                (person_id,),
            )
            return cur.fetchone()
        finally:
            cur.close()