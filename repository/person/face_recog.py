from __future__ import annotations
from typing import Optional


class FaceRecogRepository:
    def __init__(self, conn: any) -> None:
        self._conn = conn

    def insert_face_recog(self, person_id: int, predict: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO face_recog (predict, person_id)
                VALUES (%s, %s)
                """,
                (predict, person_id),
            )
        except Exception:
            raise
        finally:
            cur.close()

    def get_latest_face_recog(self, person_id: int) -> Optional[tuple]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT * FROM face_recog 
                WHERE person_id = %s 
                ORDER BY face_recog_id DESC 
                LIMIT 1
                """,
                (person_id,),
            )
            return cur.fetchone()
        finally:
            cur.close()