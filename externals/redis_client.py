import redis
from typing import Optional, Dict, Any

import core.config as config


class RedisClient:
    def __init__(
        self,
    ) -> None:
        self.host = config.Config.redis.host
        self.port = config.Config.redis.port
        self.password = config.Config.redis.password
        self.db = 0
        self.decode_responses = True
        self.client: Optional[redis.Redis] = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self.client = redis.Redis(host=self.host,
                                      port=self.port,
                                      password=self.password,
                                      decode_responses=self.decode_responses)
            self.client.ping()
            self._connected = True
            print(f"Redis connection successful")
            return True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self._connected = False
            return False

    def ensure_connection(self) -> bool:
        if not self._connected or self.client is None:
            return self.connect()
        return True

    def rpush(self, queue_name: str, payload: Dict[str, Any]) -> bool:
        if not self.ensure_connection():
            return False
        try:
            self.client.rpush(queue_name, payload)
            return True
        except Exception as e:
            print(f"RPUSH error: {e}")
            self._connected = False
            return False

    def lpop(self, queue_name: str) -> Optional[str]:
        if not self.ensure_connection():
            return None
        try:
            return self.client.lpop(queue_name)
        except Exception as e:
            print(f"LPOP error: {e}")
            self._connected = False
            return None

    def blpop(self, keys: list[str], timeout: int = 0) -> Optional[tuple[str, str]]:
        if not self.ensure_connection():
            return None
        try:
            return self.client.blpop(keys, timeout=timeout)
        except Exception as e:
            print(f"BLPOP error: {e}")
            self._connected = False
            return None
