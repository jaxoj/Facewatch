# FaceWatch

**Server-side, multi-camera face matching and watchlist monitoring system.**

FaceWatch is a modular Python-based system designed to monitor multiple camera streams, detect and track faces, compare extracted facial embeddings against a centrally managed watchlist, and generate alerts when potential matches are detected.

It is designed primarily for continuous server-side operation, supporting network cameras, distributed video sources, and headless deployments without requiring a graphical desktop environment.

FaceWatch provides a configurable processing pipeline that combines face detection, embedding extraction, vector similarity search, object tracking, temporal confirmation, and asynchronous alert delivery.

It is built with extensibility in mind, allowing individual components to be replaced or integrated with external storage systems, camera infrastructures, event-processing platforms, and notification services.

> **Important:** FaceWatch is a biometric matching and monitoring aid, not an identity verification authority. Every potential match must be reviewed by an authorized human before any action is taken. Deployments must comply with applicable privacy, biometric, surveillance, employment, and data-protection laws.

---

## Core Capabilities

- **Multi-camera monitoring:** Process multiple independent video streams concurrently.
- **Network camera integration:** Support RTSP, HTTP, video files, and locally connected cameras.
- **Face detection:** Detect faces using InsightFace models.
- **Facial embeddings:** Extract normalized 512-dimensional facial representations.
- **Watchlist management:** Maintain enrolled identities in a persistent SQLite database.
- **Vector similarity search:** Compare embeddings using NumPy or optional FAISS acceleration.
- **Face tracking:** Associate detections across consecutive frames using IoU-based tracking.
- **Temporal confirmation:** Require repeated matching results before generating an alert.
- **Asynchronous alert delivery:** Dispatch alerts without blocking camera processing.
- **Live watchlist updates:** Reload enrolled identities without restarting the monitoring service.
- **Performance controls:** Configure frame-processing intervals independently for each camera.
- **Headless operation:** Run continuously on servers without requiring a graphical interface.
- **Extensible architecture:** Replace storage, analysis, matching, and notification components through defined interfaces.

---

## System Architecture

FaceWatch follows a modular, pipeline-oriented architecture.

Each camera operates as an independent video source while sharing the configured analysis, watchlist matching, and alert infrastructure.

### Processing Pipeline

text

```text
                  CAMERA INFRASTRUCTURE
                           |
              +------------+------------+
              |            |            |
           Camera 1     Camera 2     Camera N
              |            |            |
              v            v            v
          Frame Source / Acquisition Layer
              |            |            |
              +------------+------------+
                           |
                           v
                    Frame Processing
                           |
                           v
                     Face Analyzer
                    /             \
                   v               v
             Face Detection    Embeddings
                   |               |
                   +-------+-------+
                           |
                           v
                    Watchlist Matcher
                           |
                           v
                      Face Tracker
                           |
                           v
                  Confirmation Policy
                           |
                           v
                    Alert Dispatcher
                           |
              +------------+------------+
              |            |            |
              v            v            v
           Logging     Snapshots     Webhooks
```

### Architectural Components

| Component | Responsibility |
| --- | --- |
| Source | Acquire frames from independent camera streams |
| Analyzer | Detect faces and extract facial embeddings |
| Matcher | Compare embeddings against enrolled watchlist entries |
| Tracker | Maintain face tracks across video frames |
| Confirmation Policy | Validate repeated matching results over time |
| Alert Dispatcher | Deliver confirmed events asynchronously |
| Renderer | Optionally annotate and display video frames |

The processing pipeline is implemented in `facewatch/pipeline.py`.

The application composition root is `facewatch/bootstrap.py`, which constructs and connects the concrete implementations.

Interfaces are defined through Python Protocols in `facewatch/interfaces.py`, allowing individual components to be replaced without modifying the overall pipeline.

---

## Deployment Model

FaceWatch is designed as a continuously running backend service.

It does not require a dedicated desktop application or graphical interface.

A typical deployment consists of:

text

```text
                 NETWORK CAMERA INFRASTRUCTURE
                    |       |       |
                    v       v       v
                  RTSP    RTSP    RTSP
                    |       |       |
                    +-------+-------+
                            |
                            v
                    FACEWATCH SERVER
                            |
                  +---------+---------+
                  |                   |
                  v                   v
             Face Analysis       Watchlist DB
                  |                   |
                  +---------+---------+
                            |
                            v
                     Match Processing
                            |
                            v
                      Alert Engine
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
          Monitoring    Event API     External
           Logs         / Webhooks     Systems
```

The server can operate independently of camera hardware attached directly to the machine.

Cameras are configured through their network stream addresses.

Multiple FaceWatch instances can also be deployed independently when additional processing capacity or camera isolation is required.

---

## Requirements

- Python 3.10 or newer.
- Linux server or desktop environment.
- Access to supported video sources.
- `pip` and virtual environment support.
- Sufficient CPU and memory for the configured number of streams.
- Optional NVIDIA GPU for accelerated inference.
- Optional FAISS for vector search acceleration.

Dependencies are defined in `requirements.txt`.

### Runtime Backends

**CPU execution**

Use `onnxruntime` for CPU-based inference.

**GPU execution**

Use `onnxruntime-gpu` for supported NVIDIA GPU environments.

**Vector search**

- NumPy: Default lightweight implementation.
- FAISS: Optional accelerated similarity search.

GPU acceleration should be benchmarked against the complete processing pipeline before deployment.

---

## Installation

Clone the repository:

bash

```bash
git clone <repository-url>
cd facewatch
```

Create a virtual environment:

bash

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

bash

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the configuration file:

bash

```bash
cp config.example.yaml config.yaml
```

Configure camera sources, model settings, database location, and alert destinations before starting the service.

---

## Watchlist Management

FaceWatch maintains a persistent watchlist containing enrolled identities and their corresponding facial embeddings.

### Enroll an identity

bash

```bash
python -m facewatch --config config.yaml enroll \
  --id person-001 \
  --name "Jane Doe" \
  jane-front.jpg jane-profile.jpg
```

The enrollment process:

1. Loads the supplied reference photographs.
2. Detects available faces.
3. Selects the largest usable face from each image.
4. Extracts normalized facial embeddings.
5. Combines the embeddings into a representative embedding.
6. Stores the resulting entry in SQLite.

### List enrolled identities

bash

```bash
python -m facewatch --config config.yaml list
```

### Remove an identity

bash

```bash
python -m facewatch --config config.yaml remove person-001
```

### Live watchlist updates

FaceWatch periodically reloads the watchlist database.

Newly enrolled or removed entries become available to the running monitoring process without requiring a service restart.

---

## Camera Infrastructure

FaceWatch supports multiple independent camera sources.

Supported source types include:

| Source | Example |
| --- | --- |
| RTSP cameras | `rtsp://camera/stream` |
| HTTP streams | `http://camera/video` |
| USB cameras | `0` |
| Video files | `/data/videos/recording.mp4` |

### Configure multiple cameras

yaml

```yaml
cameras:
  - id: entrance
    uri: "rtsp://camera-01/stream"
    process_every_n: 2
    enabled: true

  - id: lobby
    uri: "rtsp://camera-02/stream"
    process_every_n: 3
    enabled: true

  - id: parking
    uri: "rtsp://camera-03/stream"
    process_every_n: 2
    enabled: true
```

Each camera has:

- A unique identifier.
- An independent video source.
- An independent frame-processing interval.
- An enabled/disabled configuration.

Camera IDs are used to associate tracks, confirmation histories, and alerts with their originating sources.

### Add cameras without modifying configuration

Additional sources can be supplied through the command line:

bash

```bash
python -m facewatch run \
  --camera rtsp://camera-01/stream \
  --camera rtsp://camera-02/stream \
  --camera rtsp://camera-03/stream \
  --no-display
```

When ad-hoc cameras are supplied, they replace the configured camera list.

---

## Running the Monitoring Service

### Start configured camera infrastructure

bash

```bash
python -m facewatch --config config.yaml run
```

### Start in headless server mode

bash

```bash
python -m facewatch --config config.yaml run --no-display
```

Headless mode is recommended for production servers.

### Start with specific camera sources

bash

```bash
python -m facewatch run \
  --camera rtsp://camera-01/stream \
  --camera rtsp://camera-02/stream \
  --no-display
```

### Stop the service

Use:

bash

```bash
Ctrl+C
```

The application should shut down its camera workers and background processing components gracefully.

---

## Configuration

The complete example configuration is available in `config.example.yaml`.

Unknown configuration keys are rejected during startup to prevent silently ignored settings.

### General

| Setting | Default | Description |
| --- | --- | --- |
| `log_level` | `INFO` | Logging verbosity |
| `display` | `true` | Enable annotated video windows |
| `reload_interval_s` | `60` | Watchlist reload interval |

For server deployments, disable graphical rendering.

### Face Analysis

| Setting | Default | Description |
| --- | --- | --- |
| `model_name` | `buffalo_l` | InsightFace model pack |
| `det_size` | `[640, 640]` | Detector input dimensions |
| `ctx_id` | `0` | GPU index; `-1` selects CPU |
| `min_det_score` | `0.6` | Minimum face detection confidence |
| `min_face_px` | `40` | Minimum accepted face dimension |

### Matching

| Setting | Default | Description |
| --- | --- | --- |
| `threshold` | `0.45` | Minimum cosine similarity |
| `index` | `auto` | Search backend: `auto`, `faiss`, or `numpy` |

The similarity threshold must be calibrated using representative positive and negative samples.

### Tracking

| Setting | Default | Description |
| --- | --- | --- |
| `iou_threshold` | `0.3` | Minimum overlap for track association |
| `max_age` | `15` | Maximum unmatched frames before track expiration |

### Confirmation

| Setting | Default | Description |
| --- | --- | --- |
| `window` | `5` | Recent matching history per track |
| `min_hits` | `3` | Required matching results |
| `cooldown_s` | `30` | Alert suppression interval |

Repeated matching results reduce isolated false alerts but do not guarantee identity accuracy.

### Alert Delivery

| Setting | Default | Description |
| --- | --- | --- |
| `log` | `true` | Enable alert logging |
| `snapshot_dir` | `data/alerts` | Snapshot storage directory |
| `webhook_url` | `null` | Optional external webhook |

Webhook payloads contain:

- `camera_id`
- `entry_id`
- `name`
- `score`
- `timestamp`

---

## Performance and Resource Management

FaceWatch is designed to support different hardware configurations, from CPU-only servers to GPU-accelerated systems.

Processing intensity can be controlled independently for each camera.

### Frame Processing Intervals

The `process_every_n` setting determines how frequently a camera frame is submitted for expensive analysis.

Example:

yaml

```yaml
cameras:
  - id: entrance
    uri: "rtsp://camera-01/stream"
    process_every_n: 2

  - id: parking
    uri: "rtsp://camera-02/stream"
    process_every_n: 4
```

Higher values reduce analysis frequency and computational requirements but may increase detection latency and reduce opportunities to detect brief appearances.

### Performance Considerations

- Lower detector input dimensions reduce inference cost.
- CPU and GPU inference should be benchmarked on target hardware.
- Camera processing intervals should reflect the expected movement and traffic in each scene.
- Headless operation avoids unnecessary graphical rendering.
- FAISS can improve vector search performance for larger watchlists.
- Model selection should consider both inference speed and recognition performance.

The system should be benchmarked with the expected number of cameras and actual stream resolutions before production deployment.

---

## Storage and Persistence

The default watchlist backend uses SQLite:

text

```text
data/watchlist.db
```

Each watchlist entry stores:

- Unique identity identifier.
- Display name.
- Normalized facial embedding.
- JSON metadata.
- Creation timestamp.

The repository layer is replaceable through the `WatchlistRepository` interface.

This allows future integration with external database systems without changing the matching pipeline.

Database files and biometric data must be protected using appropriate filesystem permissions, encryption, access controls, and retention policies.

---

## Alert Processing

FaceWatch uses an asynchronous alert dispatcher with a bounded queue and background worker.

This prevents slow disk operations and webhook requests from blocking camera processing.

Supported outputs include:

- Application logs.
- JPEG snapshots.
- HTTP webhook notifications.

Snapshots follow this naming convention:

text

```text
<timestamp-milliseconds>_<camera-id>_<entry-id>.jpg
```

### Delivery Guarantees

Alert delivery is currently best-effort.

When the alert queue becomes full, new alerts are dropped and logged.

The current implementation does not provide durable event persistence, guaranteed delivery, or automatic redelivery.

Production integrations should account for these limitations.

---

## Extensibility and Integration

FaceWatch uses a Protocol-based architecture to separate business logic from implementation details.

Core components can be replaced independently:

| Interface | Purpose |
| --- | --- |
| `FrameSource` | Camera and video acquisition |
| `Analyzer` | Face detection and embedding extraction |
| `WatchlistRepository` | Identity and embedding persistence |
| `VectorIndex` | Embedding similarity search |
| `Renderer` | Optional video rendering |
| `AlertSink` | External event delivery |

This design allows future integrations with:

- Centralized database services.
- External camera management systems.
- Message brokers.
- Monitoring and observability platforms.
- Centralized event-processing infrastructure.
- Custom alerting and notification services.

The composition root in `bootstrap.py` is responsible for connecting concrete implementations.

---

## Project Structure

text

```text
facewatch/
│
├── facewatch/
│   ├── cli.py
│   ├── config.py
│   ├── domain.py
│   ├── interfaces.py
│   ├── bootstrap.py
│   ├── sources.py
│   ├── analysis.py
│   ├── enrollment.py
│   ├── repository.py
│   ├── index.py
│   ├── matcher.py
│   ├── tracking.py
│   ├── policy.py
│   ├── pipeline.py
│   ├── render.py
│   ├── alerts.py
│   └── manager.py
│
├── tests/
├── data/
├── config.example.yaml
├── requirements.txt
└── README.md
```

### Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `cli.py` | Command-line interface |
| `config.py` | Configuration loading and validation |
| `domain.py` | Shared domain objects |
| `interfaces.py` | Component protocols |
| `bootstrap.py` | Dependency composition |
| `sources.py` | Video stream acquisition |
| `analysis.py` | Face detection and embedding extraction |
| `enrollment.py` | Identity enrollment |
| `repository.py` | Watchlist persistence |
| `index.py` | Vector search implementations |
| `matcher.py` | Watchlist comparison and reload |
| `tracking.py` | Face tracking |
| `policy.py` | Temporal confirmation and cooldown |
| `pipeline.py` | Camera processing orchestration |
| `render.py` | Optional frame annotation |
| `alerts.py` | Alert dispatch and delivery |
| `manager.py` | Camera workers, lifecycle, and shutdown |

---

## Development and Testing

Run the test suite:

bash

```bash
pytest
```

Compile the project:

bash

```bash
python -m compileall -q facewatch tests
```

Tests use mock implementations of sources, analyzers, repositories, renderers, and alert sinks.

Live camera infrastructure and downloaded machine-learning models are not required for the unit tests.

---

## Current Limitations

FaceWatch is an extensible monitoring foundation, not yet a complete distributed surveillance platform.

Current limitations include:

- SQLite is intended for local deployments rather than distributed multi-server coordination.
- Webhook delivery is best-effort.
- Alert events are not durably queued.
- No built-in centralized management API is provided.
- No built-in authentication or authorization service is provided.
- Camera reconnection and fault-recovery behavior depend on the source implementation.
- Multi-server coordination and distributed watchlist synchronization are not implemented.
- Performance depends on the selected models, hardware, camera quality, and stream configuration.

These limitations should be considered when designing production deployments.

---

## Responsible Deployment

FaceWatch processes sensitive biometric information.

Operators are responsible for ensuring that deployment and operation comply with applicable laws and organizational policies.

Recommended safeguards include:

- Obtain appropriate authorization before enrolling individuals.
- Establish clear retention and deletion policies.
- Restrict access to watchlists, embeddings, snapshots, and logs.
- Secure camera credentials and network connections.
- Protect webhook endpoints.
- Audit access to biometric information.
- Require human review before acting on potential matches.
- Evaluate system performance across relevant demographic groups and operating conditions.

FaceWatch alerts indicate similarity against enrolled reference data. They must never be treated as definitive proof of identity.

---