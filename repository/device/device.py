from __future__ import annotations
from typing import Optional

class DeviceRepository:
    def __init__(self, conn: any) -> None:
        self._conn = conn

    def insert_device(
        self,
        device_name: str
    ) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO device (device_name, is_regist)
                VALUES (%s, FALSE)
                ON CONFLICT (device_name) DO NOTHING
                """,
                (device_name,),
            )
        except Exception:
            raise
        finally:
            cur.close()

    def update_is_regist(self, device_name: str, is_regist: bool) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "UPDATE device SET is_regist = %s WHERE device_name = %s",
                (is_regist, device_name),
            )
        except Exception:
            raise
        finally:
            cur.close()

    def get_device_id_db(self, device_name: str) -> Optional[tuple]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT device_id FROM device WHERE device_name = %s",
                (device_name,),
            )
            return cur.fetchone()
        finally:
            cur.close()

    def get_is_registered(self, device_name: str) -> Optional[bool]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT is_regist FROM device WHERE device_name = %s",
                (device_name,),
            )
            return cur.fetchone()
        finally:
            cur.close()