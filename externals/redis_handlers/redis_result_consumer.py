import time
from typing import List, Optional, Dict, Any

from externals.redis_client import RedisClient
from core.event_handlers import hand_in_face_event_handler, face_recog_event_handler


class RedisResultConsumer:

    def __init__(
        self,
        redis_client: RedisClient,
        poll_interval: float = 0.5,
    ) -> None:
        self.redis = redis_client
        self.poll_interval = poll_interval
        self._running = False

    def stop(self) -> None:
        self._running = False

    def run(self, queue_name: List[str]) -> None:
        self._running = True
        print("Redis consumer started")

        while self._running:

            try:

                item = self.redis.blpop(queue_name, timeout=1)

                if item is None:
                    continue
                return item.payload

            except Exception as e:
                print(f"Consumer error: {e}")
                time.sleep(self.poll_interval)

    def consume_results_from_redis_blpop(self, queue_name: List[str]) -> Optional[Dict[str, Any]]:
        self._running = True
        print("Redis consumer started")

        try:

            item = self.redis.blpop(queue_name, timeout=1)

            if item is None:
                return None
            return item.payload

        except Exception as e:
            print(f"Consumer error: {e}")

    def consume_results_from_redis_lpop(self, queue_name: List[str]) -> Optional[Dict[str, Any]]:
        self._running = True
        print("Redis consumer started")

        try:

            item = self.redis.lpop(queue_name)

            if item is None:
                return None
            return item

        except Exception as e:
            print(f"Consumer error: {e}")