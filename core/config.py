from __future__ import annotations
from dotenv import load_dotenv 
from dataclasses import dataclass
from typing import Optional
import os

load_dotenv() 

@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

@dataclass
class RedisConfig:
    host: str
    port: int
    password: Optional[str]

@dataclass
class MQTTConfig:
    broker: str
    port: int
    user: Optional[str]
    password: Optional[str]

@dataclass
class TapoCCTVConfig:
    ip: str
    user: str
    password: str
    secret: str

@dataclass
class AppConfig:
    # Camera
    camera_url: str
    camera_type: str  # webcam, rtsp

    #Device ID
    device_id: str

    # Database
    database: DatabaseConfig

    tapo_cctv: TapoCCTVConfig

    # Redis
    redis: RedisConfig

    # MQTT
    mqtt: MQTTConfig

    https_api_key: Optional[str] = None


Config: AppConfig | None = None


def init_config() -> AppConfig:
    global Config

    db_cfg = DatabaseConfig(
        host=get_env("DB_HOST", ""),
        port=get_env_int("DB_PORT", 5432),
        user=get_env("DB_USER", ""),
        password=get_env("DB_PASSWORD", ""),
        database=get_env("DB_NAME", "")
    )

    tapo_cctv_cfg = TapoCCTVConfig(
        ip=get_env("TAPO_CCTV_IP", ""),
        user=get_env("TAPO_CCTV_USER", ""),
        password=get_env("TAPO_CCTV_PASSWORD", ""),
        secret=get_env("TAPO_CCTV_SECRET", ""),
    )

    redis_cfg = RedisConfig(
        host=get_env("REDIS_HOST", ""),
        port=get_env_int("REDIS_PORT", 6379),
        password=get_env("REDIS_PASSWORD", "")
    )

    mqtt_cfg = MQTTConfig(
        broker=get_env("MQTT_BROKER", ""),
        port=get_env_int("MQTT_PORT", 1883),
        user=get_env("MQTT_USER", ""),
        password=get_env("MQTT_PASS", ""),
    )

    Config = AppConfig(
        camera_url=get_env("CAMERA_URL", ""),
        camera_type=get_env("CAMERA_TYPE", "webcam"),
        device_id=get_env("DEVICE_ID", ""),
        database=db_cfg,
        tapo_cctv=tapo_cctv_cfg,
        redis=redis_cfg,
        mqtt=mqtt_cfg,
        https_api_key=get_env("HTTPS_API_KEY", "")
    )

    return Config


def get_env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default


def get_env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default