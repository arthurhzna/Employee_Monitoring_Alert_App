from __future__ import annotations

class BehaviorRepository:
    def __init__(self, conn: any) -> None:
        self._conn = conn

    def insert_behavior(self, person_id: int, predict: str) -> None:
        """predict: 'eating' | 'drinking' | 'smoking'"""
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO behavior (predict, person_id)
                VALUES (%s, %s)
                """,
                (predict, person_id),
            )
        except Exception:
            raise
        finally:
            cur.close()