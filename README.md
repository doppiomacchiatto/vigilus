# FMV Intelligence Pipeline

A multi-stage intelligence pipeline combining fast object detection (YOLO/YOLT) with frontier vision-language model reasoning for processing airborne Full Motion Video (FMV) and satellite/aerial raster imagery. Targets high-value assets: **airplanes, vessels, and nuclear power plant facilities**.

## Pipeline Overview

```
MISB/KLV FMV Feed  ──┐
Raster Imagery     ──┤
                     ▼
              [ Stage 0: Ingest & KLV Demux ]
                     ▼
              [ Stage 1: Frame Sampling 3–5 fps ]
                     ▼
              [ Stage 2: YOLO + YOLT Tiling ]
                     ▼
              [ Stage 3: ByteTrack Tracker ]
                     ▼
              [ Stage 4: Frontier Model Verifier ]
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
    Annotated    GeoJSON/    Real-time
      Video        KML        Alerts
                              (secondary)
                     ▼
              [ Stage 6: MVS 3D Trigger ]
```

Full architecture and design decisions: [`pipeline_architecture.md`](pipeline_architecture.md)

---

## Quickstart (POC — Python)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run against a local file
```bash
# FMV / MPEG-TS feed
python fmv_pipeline.py path/to/feed.ts --output-dir output/

# Static satellite image (enable YOLT tiling)
python fmv_pipeline.py path/to/airport.png --tiling --output-dir output/

# Skip frontier model on first run to validate YOLO alone
python fmv_pipeline.py path/to/airport.png --tiling --no-frontier
```

### 4. Get a test image
- **Google Earth Pro** → navigate to `KORD`, `EGLL`, or `OMDB` → File → Save → Save Image (PNG, max res)
- **DOTA Dataset** (pre-labeled): `captain-whu.github.io/DOTA/dataset.html`
- **Copernicus / Sentinel-2** (real GeoTIFF): `dataspace.copernicus.eu`

---

## Project Structure

```
.
├── fmv_pipeline.py              # Python POC — full pipeline orchestrator
├── pipeline_architecture.md     # Architecture & design decisions
├── requirements.txt             # Python dependencies
├── Dockerfile                   # GPU-enabled container (Phase 2+)
├── .gitignore
└── src/                         # C++ hot path (Phase 2)
    ├── CMakeLists.txt
    ├── main.cpp
    ├── ingest/
    │   ├── misb_ingestor.cpp    # FFmpeg + NVDEC decode + KLV demux
    │   └── misb_ingestor.h
    ├── detect/
    │   ├── yolo_trt.cpp         # TensorRT YOLO engine wrapper (RAII)
    │   ├── yolo_trt.h
    │   ├── tile_scheduler.cpp   # YOLT chip extraction (cv::cuda)
    │   └── tile_scheduler.h
    ├── track/
    │   ├── bytetrack.cpp        # ByteTrack C++ IoU matching
    │   └── bytetrack.h
    ├── geo/
    │   ├── backproject.cpp      # PROJ/GDAL coordinate transform
    │   └── backproject.h
    ├── output/
    │   ├── video_writer.cpp     # NVENC annotated output
    │   ├── geojson_writer.cpp   # GeoJSON event sink
    │   └── kafka_emitter.cpp    # librdkafka alert producer
    └── ipc/
        ├── frontier_bridge.cpp  # Unix socket → Python sidecar
        └── frontier_bridge.h
```

---

## Deployment Phases

| Phase | Description | Key Change |
|---|---|---|
| POC | Python, local file input | Validate logic end-to-end |
| Cloud | Docker + Kafka + PostGIS | RTSP ingest, distributed output |
| Hybrid | C++ on-edge + cloud frontier | YOLO on Jetson, LLM in cloud |

---

## Configuration

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `pipeline_output` | Directory for all outputs |
| `--model` | `yolo11n.pt` | YOLO model weights path |
| `--fps` | `4.0` | Frame sample rate |
| `--tiling` | off | Enable YOLT-style frame tiling |
| `--no-frontier` | off | Skip frontier model verification |

---

## Open TODOs

- [ ] Add `RasterIngestor` class (`rasterio`-based, CRS-aware)
- [ ] Wire real `klvdata.StreamParser` for live MISB feeds
- [ ] Swap `SimpleTracker` for `supervision.ByteTrack` or `boxmot`
- [ ] Export YOLO to TensorRT `.engine` (Phase 2)
- [ ] Implement C++ `CMakeLists.txt` + TensorRT inference wrapper
- [ ] Add `VectorContextLayer` with `geopandas` spatial join
- [ ] Add Kafka producer to `AlertEmitter` (cloud phase)
- [ ] Benchmark Claude Haiku 3.5 vs GPT-4o mini on real crop data
- [ ] Llama 3.2 Vision on-container for edge Tier 1 (hybrid phase)

---

## License

MIT
