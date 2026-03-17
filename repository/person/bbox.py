from __future__ import annotations
from typing import Optional


class BboxRepository:
    def __init__(self, conn: any) -> None:
        self._conn = conn

    def insert_bbox(
        self,
        person_id: int,
        width: Optional[int],
        height: Optional[int],
    ) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO bbox (width, height, person_id)
                VALUES (%s, %s, %s)
                """,
                (width, height, person_id),
            )
        except Exception:
            raise
        finally:
            cur.close()