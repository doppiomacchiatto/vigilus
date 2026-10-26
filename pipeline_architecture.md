# FMV Intelligence Pipeline — Architecture & Design Decisions

## Overview

A multi-stage intelligence pipeline combining fast object detection (YOLO/YOLT) with deep semantic reasoning (frontier vision-language model) for processing airborne Full Motion Video (FMV) and raster/vector imagery. Targets high-value assets: **airplanes, vessels, and nuclear power plant facilities**.

---

## Pipeline Stages

| Stage | Component | Role |
|---|---|---|
| 0 | MISB/KLV Ingest | Demux MPEG-TS video + embedded geospatial metadata |
| 1 | Frame Sampler | Reduce to 3–5 fps to limit GPU load |
| 2 | YOLO + YOLT Tiling | Fast bounding box detection; tiled chips for small/aerial targets |
| 3 | ByteTrack Tracker | Persistent track IDs across frames; suppress redundant inference |
| 4 | Frontier Model Verifier | Semantic confirmation, attribute tagging, anomaly detection |
| 4b | Geospatial Back-Projection | GPS coordinate assignment per detection (KLV or rasterio CRS) |
| 5 | Output Writers | Annotated video + GeoJSON/KML events + real-time alerts |
| 6 | MVS Trigger (secondary) | 3D reconstruction when ≥15 keyframes with ≥3° gimbal parallax |

---

## Input Sources

### FMV — MISB/KLV Airborne Feed
- Container: MPEG-2 TS with embedded KLV (SMPTE ST 0601) metadata
- KLV provides per-frame: sensor lat/lon/alt, gimbal azimuth/elevation, slant range, HFOV/VFOV, platform heading, frame center coordinates
- Geospatial back-projection uses KLV frame-center + FOV linear interpolation (POC); full SMPTE ST 0601 corner-point method for production

### Raster Imagery (TODO — Task #4)
- Formats: GeoTIFF, JPEG2000, PNG, HDF5
- Library: `rasterio` for CRS-aware loading and exact geotransform back-projection
- Large scenes (satellite, aerial): YOLT-style tiling via `rasterio.windows`
- Multi-band (NIR, thermal, multispectral): band selection or false-color composite before YOLO

### Vector Context Layer
- Formats: GeoJSON, Shapefile, KML, GeoPackage
- Role: spatial join at detection time — tags confirmed detections with zone/boundary context (EEZ, restricted airspace, no-fly zones, known facility registers)
- Rule-based alert escalation for detections inside flagged polygons, without requiring frontier model call

---

## Detection Strategy

### Standard Mode
Full-frame YOLO inference. Appropriate for FMV where targets are large relative to frame.

### YOLT Tiling Mode (`--tiling`)
Overlapping 640×640 chip extraction before YOLO inference. Required when:
- Sensor altitude > ~2km (targets subtend < 32×32 pixels)
- Wide-area satellite or aerial raster scenes
- Nuclear facility overhead passes (large scene, small feature detail)

**YOLT chip parameters:** 640×640 tile size, 20% overlap, minimum chip size 160×160.

---

## Two-Tier Frontier Model Routing

Frontier model calls are gated by the ByteTrack tracker to avoid redundant inference. A track is re-verified every 15 frames or when confidence drops below threshold.

```
New or ambiguous detection crop
        │
        ▼
[Tier 1 — Hot Path]
Claude Haiku 3.5  or  GPT-4o mini
  • Structured JSON verification (class, tags, anomaly flag)
  • Target latency: < 1 second
  • Runs on every new track ID
        │
        ├── confidence > 0.85, anomaly = false → confirmed
        │
        └── confidence < 0.85  OR  anomaly = true
                │
                ▼
        [Tier 2 — Deep Analysis]
        Claude Opus 4  or  GPT-4o
          • Full reasoning with KLV/CRS context passed as text
          • Rare escalations only
          • Result triggers alert + logged to GeoJSON
```

**Edge/Hybrid deployment:** Replace Tier 1 with **Llama 3.2 Vision (11B)** running on-container — zero API dependency, no data egress, fits on a single A10G or RTX 4090.

**Model constraint:** US-company models only. Evaluated options: Anthropic (Claude), OpenAI (GPT-4o), Google (Gemini), Meta (Llama open weights).

---

## Output Formats

| Output | Format | Sink (POC → Cloud) |
|---|---|---|
| Annotated video | MP4, bounding boxes + track IDs + labels | Local file → S3/GCS |
| Geospatial events | GeoJSON / KML, one feature per confirmed detection | Local file → PostGIS |
| Real-time alerts | JSON per confirmed or anomalous detection | stdout → Kafka / Redis Streams / webhook |

---

## Secondary Requirement — MVS 3D Reconstruction

Triggered only for confirmed high-value tracks with sufficient multi-view geometry. Runs fully off the real-time hot path.

**Trigger conditions:**
- Track class: airplane, vessel, or nuclear facility
- Accumulated keyframes: ≥ 15
- Gimbal parallax delta: ≥ 3° across collected frames

**Stack:** COLMAP (SfM, sparse point cloud) → OpenMVS (dense mesh + texture). Neural alternative: Nerfstudio / 3D Gaussian Splatting for specular surfaces (aircraft fuselage, cockpit glass).

**Per-target challenges:**
- Airplanes: specular reflection on glass/metal — mask specular zones before MVS
- Vessels: dynamic water background — YOLO segmentation mask to isolate hull
- Nuclear facilities: scale + shadow occlusion — combine nadir + oblique passes

---

## C++ GPU Architecture (Phase 2+)

The Python POC validates logic. The production hot path moves to C++ for zero-copy GPU throughput.

### Design Philosophy

Only the **hot path** moves to C++. Network-bound and low-frequency components stay in Python. This is the pragmatic split — C++ earns its complexity where microseconds compound (decode, inference, IoU matching); Python wins where API latency dominates (frontier model calls, spatial joins).

Modern C++17/20 idioms used throughout: `std::unique_ptr` with custom deleters for TensorRT engine ownership, `std::shared_ptr` for shared GPU buffer lifetimes, move semantics for zero-copy frame passing between pipeline stages, `std::optional` for detection results that may be absent, and `std::queue` + `std::condition_variable` for the producer-consumer handoff between C++ and the Python sidecar.

### Zero-Copy GPU Pipeline
```
NVDEC (GPU decode)
    ↓  NV12 surface — stays in VRAM, no PCIe transfer
cv::cuda::GpuMat — resize + normalize in CUDA kernel
    ↓
TensorRT inference (GPU)
    ↓  detection tensors (GPU memory)
ByteTrack IoU matching (CPU — microseconds, small data transfer)
    ↓
Crop extraction → async queue → Python frontier sidecar
NVENC (GPU encode) → annotated output stream
```

Eliminating the GPU→CPU→GPU round-trip per frame saves ~40ms per frame at 4K/30fps — the difference between keeping up with a live feed and falling behind it.

### Docker Container Design

```dockerfile
# Base: NVIDIA CUDA development image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ── System dependencies ───────────────────────────────
RUN apt-get update && apt-get install -y \
    cmake ninja-build pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \  # FFmpeg libs
    libgdal-dev libproj-dev \                        # Geospatial
    librdkafka-dev \                                 # Kafka output
    python3.11 python3.11-dev python3-pip

# ── FFmpeg with NVDEC/NVENC ───────────────────────────
# Must build from source to enable GPU codec support
# ./configure --enable-nvdec --enable-nvenc --enable-cuda-nvcc

# ── TensorRT ─────────────────────────────────────────
# Pull from NVIDIA NGC: nvcr.io/nvidia/tensorrt:24.x-py3
# Or install via apt from NVIDIA's TensorRT repo

# ── Python sidecar ───────────────────────────────────
RUN pip install fastapi uvicorn anthropic geopandas shapely

# ── Build C++ pipeline ────────────────────────────────
COPY . /app
RUN cmake -B /app/build -S /app -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT=/usr/local/tensorrt
RUN cmake --build /app/build
```

Run with GPU access:
```bash
docker run --gpus all --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/output:/app/output \
  fmv-pipeline:latest /app/build/fmv_pipeline rtsp://feed-url
```

### C++ Hot Path — Component Breakdown

**FFmpeg `libavcodec` + NVDEC**
Decode MPEG-TS / RTSP directly to GPU memory. `AVCodecContext` configured with `AV_HWDEVICE_TYPE_CUDA` hardware acceleration context. Frames arrive as `AVFrame` with `AV_PIX_FMT_CUDA` pixel format — never touch system RAM.

**TensorRT C++ API**
YOLO model exported to `.engine` file via `trtexec`. At runtime: `IRuntime` deserializes the engine, `IExecutionContext` runs inference. Input/output bindings are pre-allocated `cudaMalloc` buffers. A `std::unique_ptr<IExecutionContext, TRTDeleter>` wraps the context with a custom deleter that calls `context->destroy()` — the canonical RAII pattern for TensorRT objects.

```cpp
// Custom deleter for TensorRT objects (modern C++ RAII)
struct TRTDeleter {
    void operator()(nvinfer1::IExecutionContext* ctx) const {
        if (ctx) ctx->destroy();
    }
};
using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter>;
```

**MITRE `libklv`**
C library for SMPTE ST 0601 KLV parsing. Replaces the Python `klvdata` stub. Parses the MPEG-TS data PID in a dedicated thread, writes `KLVFrame` structs into a lock-free ring buffer keyed by PTS.

**ByteTrack C++**
Standalone C++ port of ByteTrack. IoU matching runs in microseconds on CPU — the detection set per frame is small enough that GPU acceleration here would cost more in transfer overhead than it saves.

**PROJ / GDAL C API**
`proj_create_crs_to_crs()` for coordinate transforms. Called once per detection to back-project pixel coordinates to WGS-84. For raster inputs, `GDALDataset::GetGeoTransform()` provides the affine transform directly.

**`librdkafka`**
C++ Kafka producer for the alert output stream. `RdKafka::Producer::produce()` is non-blocking — detection events are enqueued and flushed asynchronously without stalling the hot path.

### Python Sidecar — What Stays in Python

| Component | Why Python | Library |
|---|---|---|
| Frontier model calls (Tier 1 + 2) | Network-bound; API latency >> language overhead | `anthropic` / `openai` |
| REST control plane | Config, health checks, pipeline control | `FastAPI` |
| Vector spatial joins | `geopandas` is mature and fast enough at detection rates | `geopandas`, `shapely` |
| MVS job submission | Infrequent; subprocess call to COLMAP | `subprocess` |

The C++ hot path communicates with the Python sidecar via a **Unix domain socket** (fastest IPC on the same host) carrying length-prefixed protobuf or MessagePack messages. Each message contains the JPEG-encoded crop + track ID + KLV metadata. The sidecar deserializes, calls the frontier model, and writes the result back.

### CMakeLists.txt Structure (Phase 2 artifact — TODO)

```
CMakeLists.txt
src/
  main.cpp              # pipeline orchestrator, thread launch
  ingest/
    misb_ingestor.cpp   # FFmpeg + NVDEC decode + KLV demux thread
  detect/
    yolo_trt.cpp        # TensorRT engine wrapper (RAII)
    tile_scheduler.cpp  # YOLT chip extraction (cv::cuda)
  track/
    bytetrack.cpp       # ByteTrack C++ IoU matching
  geo/
    backproject.cpp     # PROJ/GDAL coordinate transform
  output/
    video_writer.cpp    # NVENC annotated output
    geojson_writer.cpp  # GeoJSON event sink
    kafka_emitter.cpp   # librdkafka alert producer
  ipc/
    frontier_bridge.cpp # Unix socket → Python sidecar
```

---

## Deployment Progression

### POC
Single machine, MPEG-TS or static image file input, synchronous pipeline, all stages in one Python process. Goal: validate KLV demux, back-projection math, tracker logic, frontier model prompts, GeoJSON schema.

### Cloud
- Replace file input with RTSP/RTP ingestion
- Containerize each stage (Docker + Kubernetes)
- Message broker (Kafka or Redis Streams) between stages
- Outputs: S3/GCS for annotated video, PostGIS for GeoJSON events
- Frontier model via managed API endpoint

### Hybrid
- YOLO + ByteTrack on-edge (Jetson Orin NX or equivalent)
- Only confirmed/ambiguous crops + KLV snippets uplinked to cloud
- Frontier model + MVS remain cloud-side
- Alerts stream back to ground station via low-bandwidth side channel
- Tier 1 frontier model replaced with Llama 3.2 Vision on-container

---

## Imagery Sources for Testing

| Source | Resolution | Format | Notes |
|---|---|---|---|
| Google Earth Pro | ~0.3m (varies) | PNG/JPEG | Manual export; fastest for POC |
| DOTA Dataset | High-res | PNG + labels | Pre-labeled airplane/ship/facility bounding boxes; recommended for YOLO validation |
| Copernicus / Sentinel-2 | 10m | GeoTIFF | Free, global, native CRS metadata; good for rasterio integration testing |
| Planet Labs API | 3m | GeoTIFF | Programmatic, daily revisit, Python SDK |
| USGS Earth Explorer | Varies | GeoTIFF | Free account required |

### Google Earth Pro — Step-by-Step POC Export

1. Download free at `earth.google.com/web/` → Desktop version
2. In the search bar navigate to a dense airport:
   - `KORD` — Chicago O'Hare (high cargo + commercial density)
   - `EGLL` — London Heathrow (dense stand parking, good parallax)
   - `OMDB` — Dubai International (wide aprons, clear atmosphere)
   - `VHHH` — Hong Kong (tight ramp, good for small-object testing)
3. Zoom to approximately 500m eye altitude for best aircraft visibility
4. **File → Save → Save Image** — set max resolution (4800×2717)
5. Export as PNG for lossless quality
6. Run through pipeline: `python fmv_pipeline.py airport.png --tiling --no-frontier`

### DOTA Dataset — Recommended for YOLO Validation

Purpose-built satellite imagery benchmark with pre-labeled bounding boxes for airplanes, ships, storage tanks, harbors, and more. Use this to measure detection accuracy before touching live data.

- Download: `captain-whu.github.io/DOTA/dataset.html`
- YOLO models pre-trained on DOTA available on Hugging Face (search `DOTA YOLO`)
- Format: PNG images + DOTA-format `.txt` label files (convertible to YOLO format with a one-liner)

### Copernicus Open Access Hub — Free GeoTIFF with Native CRS

Best source for testing the `RasterIngestor` + `CRSBackProjector` with real embedded coordinate metadata.

- Register free at `dataspace.copernicus.eu`
- Search for Sentinel-2 L2A (atmospherically corrected) over any major airport
- Download the 10m resolution bands (B02, B03, B04 = RGB)
- File arrives as a GeoTIFF with full EPSG projection metadata — exactly what `rasterio` needs

### OpenAerialMap — High-Resolution Open License

- Browse and download at `openaerialmap.org`
- Some tiles are sub-0.5m resolution, CC-licensed, no registration required
- Good for finding high-quality airport coverage contributed by mapping NGOs

### Planet Labs API — Programmatic Integration

For wiring automated imagery pull into the pipeline:

```python
import planet

async def fetch_airport_chip(lat, lon, date_str, api_key):
    """Pull a 3m resolution GeoTIFF chip over any airport bounding box."""
    async with planet.Session(auth=planet.Auth.from_key(api_key)) as session:
        client = session.client("data")
        # Define AOI around airport centroid (roughly 5km box)
        aoi = {
            "type": "Point",
            "coordinates": [lon, lat]
        }
        # Search for most recent clear Skysat or PlanetScope scene
        # Export as GeoTIFF → feed directly to RasterIngestor
```

Free trial at `planet.com/explorer` — includes limited download credits.

---

## Open TODOs

- [ ] **Task #4** — Add `RasterIngestor` class: `rasterio`-based loader, `CRSBackProjector` to replace KLV math, integration with existing `tile_frame()` tiling
- [ ] Wire real `klvdata.StreamParser` into `MISBIngestor._parse_klv()` for live MISB feeds
- [ ] Swap `SimpleTracker` for `supervision.ByteTrack` or `boxmot` for production tracking
- [ ] Export YOLO model to TensorRT `.engine` and benchmark vs PyTorch (Phase 2)
- [ ] Implement `CMakeLists.txt` and TensorRT C++ inference wrapper (Phase 2)
- [ ] Add `VectorContextLayer` with `geopandas` point-in-polygon spatial join
- [ ] Add Kafka producer to `AlertEmitter` for cloud deployment
- [ ] Benchmark Claude Haiku 3.5 vs GPT-4o mini on real crop data for Tier 1 model selection
- [ ] Implement Llama 3.2 Vision on-container for edge Tier 1 replacement
