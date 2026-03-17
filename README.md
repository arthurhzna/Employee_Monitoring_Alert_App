# Employee Monitoring Alert App

> Real-time edge AI surveillance system for operator and workplace safety monitoring. Processes live video feeds to detect drowsiness, distraction, hand-to-face gestures, and risky behaviors, then triggers immediate multi-channel alerts.

---

## Overview

The system is built around a **layered, dependency-injected architecture**: a thin pipeline orchestrator delegates to composable feature modules, each of which owns its own ML model(s) and emits typed events to a thread-safe event bus. Persistence, alerting, and async ML tasks are handled by subscribers — keeping the hot path free of I/O.

---

## Key Capabilities

| Capability | Technique | Trigger |
|---|---|---|
| **Person detection & tracking** | YOLOv11n + ByteTrack | Every frame |
| **Face detection** | MediaPipe FaceDetection (model 0) | Per tracked person |
| **Face direction / attention** | Landmark distance ratios (MediaPipe FaceMesh, 468 pts) | Per detected face |
| **Drowsiness** | Eye Aspect Ratio (EAR) < 0.15 + yawn confirmation | Per detected face |
| **Yawning** | Mouth Aspect Ratio (MAR) via face mesh landmarks | Per detected face |
| **Hand-to-face contact** | Hand landmark bounding box ∩ face bounding box | Per detected face |
| **Dwell time** | Cumulative "Looking" / "Not Looking" time per track ID | Per person session |
| **Face recognition** | InsightFace `buffalo_l` (async microservice) | On first "Looking" frame |
| **Behavior classification** | Qwen3-VL-2B VLM (async microservice) | On hand-in-face trigger |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          alert_app (main process)                   │
│                                                                     │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐   │
│  │  Input Layer │    │              Pipeline                   │   │
│  │              │    │  core/pipeline.py                       │   │
│  │  RTSP Camera │───►│  • camera.read()                        │   │
│  │  (threaded)  │    │  • resize_frame(640×480)                │   │
│  │  Webcam      │    │  • fps_counter.update()                 │   │
│  └──────────────┘    │  • person_analysis_feature.process()   │   │
│                      │  • display.show()                       │   │
│                      └──────────────┬──────────────────────────┘   │
│                                     │                               │
│                      ┌──────────────▼──────────────────────────┐   │
│                      │        PersonAnalysisFeature             │   │
│                      │  features/person_analysis_feature.py    │   │
│                      │                                          │   │
│                      │  ┌─────────────────────────────────┐    │   │
│                      │  │  YOLOv11n + ByteTrack            │    │   │
│                      │  │  inference/yolo_model.py         │    │   │
│                      │  │  • detect class=0 (person)       │    │   │
│                      │  │  • assign persistent track_id    │    │   │
│                      │  │  • conf threshold: 0.5           │    │   │
│                      │  └──────────────┬──────────────────┘    │   │
│                      │                 │ per-person crop        │   │
│                      │                 │ (expand_factor=0.2)    │   │
│                      └─────────────────┼────────────────────────┘   │
│                                        │                             │
│                      ┌─────────────────▼────────────────────────┐   │
│                      │          FaceAnalysisFeature              │   │
│                      │  features/face_analysis_feature.py        │   │
│                      │                                           │   │
│                      │  ① Redis lpop → face_recog_results        │   │
│                      │  ② Redis lpop → behavior_results          │   │
│                      │                                           │   │
│                      │  ┌──────────────────────────────────┐    │   │
│                      │  │  MediaPipe FaceDetection          │    │   │
│                      │  │  • min_confidence: 0.9            │    │   │
│                      │  │  • returns relative bounding box  │    │   │
│                      │  └──────────────┬───────────────────┘    │   │
│                      │                 │ face crop               │   │
│                      │  ┌──────────────▼───────────────────┐    │   │
│                      │  │  MediaPipe FaceMesh               │    │   │
│                      │  │  • 468 landmarks                  │    │   │
│                      │  │  • refine_landmarks=True          │    │   │
│                      │  └──────────────┬───────────────────┘    │   │
│                      │                 │                         │   │
│                      │  ┌──────────────▼───────────────────┐    │   │
│                      │  │  MediaPipe Hands                  │    │   │
│                      │  │  • max_num_hands=2                │    │   │
│                      │  │  • run on person crop             │    │   │
│                      │  └──────────────┬───────────────────┘    │   │
│                      │                 │                         │   │
│                      │  ┌──────────────▼───────────────────┐    │   │
│                      │  │  Feature Modules                  │    │   │
│                      │  │  ├─ FaceDirectionFeature          │    │   │
│                      │  │  │    landmark distance ratios    │    │   │
│                      │  │  ├─ FaceDwellTimeFeature          │    │   │
│                      │  │  │    cumulative attention time   │    │   │
│                      │  │  ├─ DrowsinessFeature (EAR)       │    │   │
│                      │  │  ├─ YawningFeature (MAR)          │    │   │
│                      │  │  └─ HandInFaceFeature             │    │   │
│                      │  │       landmark ∩ face bbox        │    │   │
│                      │  └──────────────────────────────────┘    │   │
│                      └──────────────────────────────────────────┘   │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      EventBus (thread-safe pub/sub)           │  │
│  │  events/event_bus.py  •  threading.Lock  •  typed events      │  │
│  │                                                               │  │
│  │  HAND_IN_FACE_DETECTED  ──► save image → rpush behavior_queue │  │
│  │  FACE_RECOG_DETECTED    ──► save image → rpush face_recog_queue│  │
│  │  DROWSINESS_DETECTED    ──► Tapo CCTV speaker (non-blocking)  │  │
│  │  PERSON_DISAPPEARED     ──► PostgreSQL (person+bbox+dwell)    │  │
│  │  FACE_RECOG_RESULT      ──► PostgreSQL (face_recog)           │  │
│  │  BEHAVIOR_RESULT        ──► PostgreSQL (behavior)             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Async Microservices

Heavy ML inference (face recognition, VLM behavior classification) runs in **separate processes** to avoid blocking the real-time pipeline. Communication is via Redis queues.

```
┌────────────────────────────────────────────────────────────────────┐
│                        Redis (shared)                              │
│                                                                    │
│  alert_app ──rpush──► behavior_queue ──lpop──► behavior_predict   │
│  alert_app ──rpush──► face_recog_queue ──lpop──► face_recog_predict│
│                                                                    │
│  behavior_predict ──rpush──► behavior_results_<device_id>         │
│  face_recog_predict ──rpush──► face_recog_results_<device_id>     │
│                                    └──lpop──► alert_app           │
└────────────────────────────────────────────────────────────────────┘

face_recog_predict/
  • FastAPI service
  • InsightFace buffalo_l (ArcFace)
  • Returns: { uuid, predict: "name | unknown" }

behavior_predict/
  • Qwen3-VL-2B / Qwen2-VL-2B (Vision Language Model)
  • Returns: { uuid, predict: "eating | drinking | smoking | none" }
```

---

## Multi-Camera Deployment

Each `alert_app` instance handles exactly one camera. Multiple instances share the same PostgreSQL and Redis, with device isolation via `DEVICE_ID`.

```
┌──────────────────────────────────────────────────────────────────┐
│  alert_app #1  (DEVICE_ID=cam-01, CAMERA_URL=rtsp://...)         │
│  alert_app #2  (DEVICE_ID=cam-02, CAMERA_URL=rtsp://...)         │
│  alert_app #3  (DEVICE_ID=cam-03, CAMERA_URL=rtsp://...)  ───────┼──► PostgreSQL
│  alert_app #4  (DEVICE_ID=cam-04, CAMERA_URL=rtsp://...)         │
│  alert_app #5  (DEVICE_ID=cam-05, CAMERA_URL=rtsp://...)         │
└──────────────────────────────────────────────────────────────────┘
         │                                    ▲
         └──────────────► Redis ◄─────────────┘
                    face_recog_predict
                    behavior_predict
```

**Concurrency safety:**
- `ThreadedConnectionPool` — each background worker thread gets its own DB connection
- `DataStore.atomic()` — all related INSERTs execute in a single atomic transaction
- `ON CONFLICT DO NOTHING` — safe concurrent device/person registration across instances

---

## Data Flow (per frame)

```
RTSP frame
    │
    ▼ resize to 640×480
    │
    ▼ YOLOv11n + ByteTrack  →  [track_id, bbox] per person
    │
    ├─ for each tracked person:
    │       │
    │       ▼ crop person region (+20% expand)
    │       │
    │       ▼ MediaPipe FaceDetection  →  face bbox (relative)
    │       │
    │       ├─ for each detected face:
    │       │       │
    │       │       ▼ crop face region
    │       │       │
    │       │       ├─ FaceMesh (468 landmarks)
    │       │       │       ├─ FaceDirectionFeature  →  "Looking" | "Not Looking" | "Left" | "Right"
    │       │       │       ├─ DrowsinessFeature     →  EAR < 0.15
    │       │       │       └─ YawningFeature        →  MAR threshold
    │       │       │
    │       │       └─ HandMesh (on person crop)
    │       │               └─ HandInFaceFeature     →  landmark ∩ face bbox overlap
    │       │
    │       └─ FaceDwellTimeFeature  →  accumulate looking/not-looking seconds
    │
    ├─ on "Looking" (first time):    publish FACE_RECOG_DETECTED
    ├─ on hand-in-face (rising edge): publish HAND_IN_FACE_DETECTED
    ├─ on drowsiness + yawn (rising edge): publish DROWSINESS_DETECTED
    └─ on person disappear:          publish PERSON_DISAPPEARED
```

---

## Project Structure

```
alert_app/
├── main.py                        # Entry point — wires all components
│
├── core/
│   ├── config.py                  # Typed config from .env (dataclasses)
│   ├── container.py               # Dependency injection container
│   ├── database.py                # PostgreSQL ThreadedConnectionPool + migrations
│   ├── event_handlers.py          # Event handler factories (closures)
│   ├── model.py                   # PersonData dataclass (per-track state)
│   ├── pipeline.py                # Main loop: read → resize → process → display
│   ├── state_manager.py           # Thread-safe device state (device_id, is_registered)
│   └── worker.py                  # Background ThreadPoolExecutor for I/O tasks
│
├── input/
│   ├── base_camera.py             # Abstract camera interface
│   ├── rtsp_camera.py             # RTSP reader (background thread, latest-frame buffer)
│   └── webcam_camera.py           # OpenCV webcam reader
│
├── output/
│   └── display.py                 # OpenCV imshow window
│
├── preprocessing/
│   ├── resize.py                  # resize_frame(640×480)
│   └── crop.py                    # crop_person_with_expand, crop_face
│
├── inference/
│   ├── base_model.py              # Abstract model interface (load/predict/release)
│   ├── yolo_model.py              # YOLOv11n wrapper (ultralytics + ByteTrack)
│   ├── insightface_model.py       # InsightFace buffalo_l (used in face_recog_predict/)
│   ├── qwen3_vl_model.py          # Qwen3-VL-2B VLM (used in behavior_predict/)
│   ├── qwen2_vl_model.py          # Qwen2-VL-2B VLM (alternative)
│   ├── vit_transformer_model.py   # ViT classifier (alternative)
│   └── mediapipe/
│       ├── base_mediapipe.py      # Base MediaPipe wrapper
│       ├── face_detect.py         # FaceDetection (model_selection=0)
│       ├── face_mesh.py           # FaceMesh (468 landmarks, refine=True)
│       └── hand_mesh.py           # Hands (max_num_hands=2)
│
├── features/
│   ├── person_analysis_feature.py # Top-level: YOLO → per-person dispatch
│   ├── face_analysis_feature.py   # Face pipeline: detect → mesh → features → events
│   ├── hand_in_face_feature.py    # Hand landmark ∩ face bbox overlap
│   ├── face/
│   │   ├── direction_feature.py   # Gaze direction from mesh landmark ratios
│   │   ├── dwelltime_feature.py   # Attention dwell time accumulator
│   │   ├── drowsiness_feature.py  # EAR-based eye closure detection
│   │   └── yawning_feature.py     # MAR-based yawn detection
│   └── behavior/
│       ├── base_behavior.py
│       ├── eating_validate.py     # Eating heuristic (pre-VLM gate)
│       ├── drinking_validate.py   # Drinking heuristic
│       └── smoking_validate.py    # Smoking heuristic
│
├── events/
│   ├── base_event.py
│   ├── event_bus.py               # Thread-safe pub/sub (threading.Lock)
│   ├── drowsiness_event.py
│   ├── hand_in_face_event.py
│   ├── face_recog_event.py
│   ├── face_recog_result_event.py
│   ├── behavior_result_event.py
│   └── person_disappeared_event.py
│
├── repository/
│   ├── data_store.py              # DataStore — atomic() + query() entry point
│   ├── tx.py                      # Tx — repository accessor inside a transaction
│   ├── device/
│   │   └── device.py              # DeviceRepository (insert, get_id, get_is_registered)
│   └── person/
│       ├── person.py              # PersonRepository
│       ├── bbox.py                # BboxRepository
│       ├── dwelltime.py           # DwelltimeRepository
│       ├── face_recog.py          # FaceRecogRepository
│       ├── behavior.py            # BehaviorRepository
│       └── drowsiness.py         # DrowsinessRepository
│
├── migrations/                    # yoyo migration scripts (SQL)
│
├── externals/
│   ├── redis_client.py            # Redis connection wrapper (rpush/lpop)
│   ├── redis_handlers/
│   │   └── redis_result_consumer.py     # lpop consumer for async ML results
│   ├── mqtt_client.py             # MQTT client (currently disabled)
│   ├── mqtt_handlers/
│   │   ├── base_handler.py              # Abstract MQTT message handler
│   │   └── device_registration_handler.py  # Handles device registration via MQTT
│   ├── tapo_cctv_speaker_client.py      # Tapo CCTV audio alert (streamovat)
│   ├── speaker_tapo_cctv_handlers/
│   │   ├── http_audio_session.py        # HTTP audio streaming session
│   │   ├── resampling_audio.py          # Audio resampling utility
│   │   ├── hfix.py                      # Audio format helper
│   │   ├── g.py                         # Audio codec helper
│   │   └── util.py                      # Shared utilities
│   ├── websocket_client.py
│   └── http_client.py
│
├── tracking/
│   └── costum_tracker.py          # Stub (ByteTrack used via ultralytics)
│
├── models/
│   └── yolo/universal/yolo11n/
│       └── yolo11n.pt             # YOLOv11n weights (~6 MB)
│
├── audio/
│   └── fokus.wav                  # Alert audio played via Tapo speaker
│
├── data/
│   ├── hand_in_face/              # Saved frames for behavior inference
│   └── face_recog/                # Saved frames for face recognition
│
└── utils/
    ├── fps_counter.py
    ├── image_utils.py             # draw_person_overlay, write_image
    └── tensor_utils.py            # get_tensor_value (YOLO tensor helpers)
```

---

## Database Schema

```
device
  └── id, device_name, is_registered

person
  └── id, uuid, device_id (FK), face_recog, eating/drinking/smoking flags,
      timestamp_start, timestamp_end

bbox
  └── id, person_id (FK), width, height

dwelling_time
  └── id, person_id (FK), dwelling_looking (s), dwelling_not_looking (s)

face_recog
  └── id, person_id (FK), predict (name | unknown)

behavior
  └── id, person_id (FK), predict (eating | drinking | smoking | none)

drowsiness
  └── id, person_id (FK), timestamp
```

All writes go through `DataStore.atomic()` — a context-managed transaction that acquires a pooled connection, runs all repository calls, and commits or rolls back atomically.

---

## Infrastructure (Docker)

```bash
docker-compose up -d
```

| Service | Port | Description |
|---|---|---|
| PostgreSQL | `5433` | Main database |
| Redis | `6379` | Async result queue |
| pgAdmin | `8888` | DB management UI |
| RedisInsight | `5540` | Redis management UI |

---

## Requirements

- Python 3.10+
- CUDA 12.4 compatible GPU (recommended for YOLO inference)
- PostgreSQL 15+
- Redis 7+
- (Optional) Tapo CCTV camera with speaker
- (Optional) MQTT broker

---

## Configuration

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `DEVICE_ID` | Unique identifier for this device/camera instance |
| `CAMERA_URL` | RTSP stream URL or `0` for webcam |
| `CAMERA_TYPE` | `rtsp` or `webcam` |
| `DB_HOST` | PostgreSQL hostname |
| `DB_PORT` | PostgreSQL port (default `5433`) |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password |
| `DB_NAME` | PostgreSQL database name |
| `REDIS_HOST` | Redis hostname |
| `REDIS_PORT` | Redis port (default `6379`) |
| `REDIS_PASSWORD` | Redis password |
| `TAPO_CCTV_IP` | IP address of the Tapo CCTV camera |
| `TAPO_CCTV_USER` | Tapo camera username |
| `TAPO_CCTV_PASSWORD` | Tapo camera password |
| `TAPO_CCTV_SECRET` | Tapo camera cloud secret |
| `MQTT_BROKER` | MQTT broker hostname |
| `MQTT_PORT` | MQTT broker port |
| `MQTT_USER` | MQTT username |
| `MQTT_PASS` | MQTT password |

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ultralytics` | 8.3.49 | YOLOv11n person detection + ByteTrack tracking |
| `mediapipe` | 0.10.18 | Face detection, face mesh (468 pts), hand mesh |
| `torch` / `torchvision` | 2.5.1 | Deep learning backend (YOLO GPU inference) |
| `opencv-python` | 4.10.0.84 | Frame capture, image processing, display |
| `psycopg2-binary` | — | PostgreSQL adapter with ThreadedConnectionPool |
| `yoyo-migrations` | — | SQL schema migration runner |
| `redis` | — | Async result queue (rpush/lpop) |
| `python-dotenv` | — | `.env`-based typed configuration |
| `pytapo` | — | Tapo CCTV camera speaker API |
| `numpy` | — | EAR/MAR landmark distance calculations |
| `insightface` | — | ArcFace face recognition (microservice) |
| `transformers` | 5.2.0 | Qwen3-VL-2B VLM (behavior microservice) |

---

## Related Services

| Service | Description |
|---|---|
| **`face_recog_predict/`** | FastAPI + InsightFace `buffalo_l`. Consumes `face_recog_queue`, returns identity labels to `face_recog_results_<device_id>` |
| **`behavior_predict/`** | Qwen3-VL-2B VLM. Consumes `behavior_queue`, returns `eating / drinking / smoking / none` to `behavior_results_<device_id>` |
