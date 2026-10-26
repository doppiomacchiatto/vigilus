"""
FMV Intelligence Pipeline
MISB/KLV Airborne Feed → YOLO Detection → Frontier Model Verification
→ Annotated Video + GeoJSON Events + Real-time Alerts

POC entry point: process a local MPEG-TS file end-to-end.
Cloud/Hybrid variants swap the Ingestor for an RTSP/RTP reader and
replace the OutputWriter sinks with Kafka + S3 + PostGIS.

Requirements (install before running):
    pip install ultralytics klvdata geojson aiohttp opencv-python-headless numpy
    pip install anthropic          # for frontier model calls (swap for openai if preferred)

For YOLT-style tiling on small/aerial targets, the `tile_frame` helper
slices the frame before passing to the YOLO detector — no separate model needed.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import geojson
from ultralytics import YOLO

# ── KLV demux (klvdata library) ───────────────────────────────────────────────
# klvdata parses SMPTE ST 0601 KLV packets extracted by ffmpeg or direct MPEG-TS demux.
try:
    import klvdata  # pip install klvdata
    KLV_AVAILABLE = True
except ImportError:
    KLV_AVAILABLE = False
    print("[WARNING] klvdata not installed — KLV metadata will be stubbed.")

# ── Frontier model client (Anthropic Claude) ──────────────────────────────────
try:
    import anthropic
    FRONTIER_CLIENT = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    FRONTIER_AVAILABLE = True
except ImportError:
    FRONTIER_AVAILABLE = False
    print("[WARNING] anthropic SDK not installed — frontier verification disabled.")


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class KLVFrame:
    """Geospatial + sensor metadata for a single video frame."""
    pts: int                        # presentation timestamp (microseconds)
    sensor_lat: float = 0.0         # degrees
    sensor_lon: float = 0.0         # degrees
    sensor_alt: float = 0.0         # metres MSL
    platform_heading: float = 0.0   # degrees true north
    gimbal_az: float = 0.0          # degrees
    gimbal_el: float = -90.0        # degrees (nadir = -90)
    slant_range: float = 0.0        # metres
    hfov: float = 5.0               # horizontal field of view, degrees
    vfov: float = 3.0               # vertical field of view, degrees
    frame_center_lat: float = 0.0
    frame_center_lon: float = 0.0


@dataclass
class Detection:
    """Single confirmed object detection with geospatial context."""
    track_id: int
    label: str
    confidence: float
    bbox_xyxy: list[float]          # pixel coords in original frame
    geo_lat: float = 0.0
    geo_lon: float = 0.0
    geo_alt: float = 0.0
    frontier_verified: bool = False
    frontier_tags: list[str] = field(default_factory=list)
    pts: int = 0


# =============================================================================
# Stage 0 — Ingest & KLV Demux
# =============================================================================

class MISBIngestor:
    """
    Reads an MPEG-TS file (or RTSP URL for cloud/hybrid) and yields
    (frame_bgr, KLVFrame) pairs at the requested sample rate.

    For live RTSP, swap Path(source) for the RTSP URL string; cv2.VideoCapture
    handles both transparently.

    KLV is extracted from the MPEG-TS data stream via klvdata. If the file
    has no embedded KLV, a zeroed KLVFrame stub is returned.
    """

    def __init__(self, source: str | Path, sample_fps: float = 4.0):
        self.source = str(source)
        self.sample_fps = sample_fps
        self._cap = cv2.VideoCapture(self.source)
        native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._step = max(1, round(native_fps / sample_fps))
        self._frame_idx = 0

    def __iter__(self) -> Generator[tuple[np.ndarray, KLVFrame], None, None]:
        klv_meta = KLVFrame(pts=0)  # updated each time KLV packet is parsed

        while True:
            ok, frame = self._cap.read()
            if not ok:
                break

            self._frame_idx += 1
            if self._frame_idx % self._step != 0:
                continue

            pts = int(self._cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)
            klv_meta = self._parse_klv(pts) or klv_meta
            yield frame, klv_meta

    def _parse_klv(self, pts: int) -> Optional[KLVFrame]:
        """
        Stub: in a real MISB feed, extract the KLV data stream alongside video.
        With ffmpeg you can pipe: ffmpeg -i input.ts -map 0:d -c copy data.klv
        then parse with klvdata.  Here we return a zeroed frame so the rest of
        the pipeline can run on plain MP4 test files.
        """
        if not KLV_AVAILABLE:
            return KLVFrame(pts=pts)
        # TODO: wire in actual klvdata.StreamParser for real MISB feeds
        return KLVFrame(pts=pts)

    def release(self):
        self._cap.release()


# =============================================================================
# Stage 1 — YOLT-style Tiling (optional, for small/aerial targets)
# =============================================================================

def tile_frame(
    frame: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """
    Slice the frame into overlapping chips for YOLT-style multi-scale detection.
    Returns list of (chip, (x_offset, y_offset)) so detections can be
    re-projected back to full-frame pixel coordinates.

    Only needed when targets are small relative to the full frame (e.g. >2km
    sensor altitude with nuclear facility or vessel in frame).
    """
    h, w = frame.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            chip = frame[y:y2, x:x2]
            if chip.shape[0] < tile_size // 4 or chip.shape[1] < tile_size // 4:
                continue
            tiles.append((chip, (x, y)))
    return tiles


# =============================================================================
# Stage 2 — YOLO Detector
# =============================================================================

class YOLODetector:
    """
    Wraps Ultralytics YOLO.  Labels of interest are filtered to the three
    high-value target classes; everything else is discarded before passing
    to the tracker.
    """

    HIGH_VALUE_LABELS = {"airplane", "boat", "ship", "vessel", "nuclear", "plant"}

    def __init__(self, model_path: str = "yolo11n.pt", conf: float = 0.35, use_tiling: bool = False):
        self.model = YOLO(model_path)
        self.conf = conf
        self.use_tiling = use_tiling

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts: {label, confidence, xyxy, chip_offset}.
        When tiling is enabled, offsets are added back to xyxy coords.
        """
        results = []

        frames_to_process: list[tuple[np.ndarray, tuple[int, int]]] = (
            tile_frame(frame) if self.use_tiling
            else [(frame, (0, 0))]
        )

        for chip, (ox, oy) in frames_to_process:
            preds = self.model(chip, conf=self.conf, verbose=False)[0]
            for box in preds.boxes:
                label = preds.names[int(box.cls[0])]
                if not any(kw in label.lower() for kw in self.HIGH_VALUE_LABELS):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                results.append({
                    "label": label,
                    "confidence": float(box.conf[0]),
                    "xyxy": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
                })

        return results


# =============================================================================
# Stage 3 — ByteTrack-style Temporal Tracker (lightweight stub)
# =============================================================================

class SimpleTracker:
    """
    Minimal IoU-based tracker sufficient for the POC.
    Replace with `supervision.ByteTrack` or `boxmot` for production.
    Each active track carries a verification_count to suppress redundant
    frontier model calls once a track is confirmed.
    """

    def __init__(self, iou_threshold: float = 0.4, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[Detection]:
        """Match detections to existing tracks; assign new IDs where needed."""
        matched = []
        unmatched_dets = list(detections)

        for track_id, track in list(self._tracks.items()):
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(unmatched_dets):
                iou = self._iou(track["xyxy"], det["xyxy"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= self.iou_threshold:
                det = unmatched_dets.pop(best_idx)
                track.update({"xyxy": det["xyxy"], "label": det["label"],
                               "confidence": det["confidence"], "age": 0})
                matched.append((track_id, track))
            else:
                track["age"] += 1
                if track["age"] > self.max_age:
                    del self._tracks[track_id]

        for det in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {**det, "age": 0, "verification_count": 0}
            matched.append((tid, self._tracks[tid]))

        return [
            Detection(
                track_id=tid,
                label=t["label"],
                confidence=t["confidence"],
                bbox_xyxy=t["xyxy"],
                pts=0,
            )
            for tid, t in matched
        ]

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


# =============================================================================
# Stage 4 — Frontier Model Semantic Verifier
# =============================================================================

VERIFICATION_INTERVAL = 15  # re-verify a track every N frames

TARGET_PROMPT = """
You are a remote sensing analyst reviewing a cropped image patch from an airborne sensor.
Confirm whether the object is one of: airplane, vessel, nuclear power plant facility.
Respond ONLY with valid JSON in this exact schema:
{
  "confirmed": true | false,
  "class": "airplane" | "vessel" | "nuclear_facility" | "unknown",
  "confidence": 0.0-1.0,
  "tags": ["list", "of", "attribute", "tags"],
  "anomaly": true | false,
  "notes": "brief free-text note"
}
"""


def verify_with_frontier(crop_bgr: np.ndarray) -> dict:
    """
    Encode the crop as JPEG and send to the frontier vision model.
    Returns parsed JSON dict; on failure returns a safe default.
    """
    if not FRONTIER_AVAILABLE:
        return {"confirmed": True, "class": "unknown", "confidence": 0.5,
                "tags": [], "anomaly": False, "notes": "frontier unavailable"}

    _, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode()

    try:
        response = FRONTIER_CLIENT.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/jpeg",
                                                  "data": b64}},
                    {"type": "text", "text": TARGET_PROMPT},
                ],
            }],
        )
        return json.loads(response.content[0].text)
    except Exception as exc:
        print(f"[FRONTIER] Error: {exc}")
        return {"confirmed": True, "class": "unknown", "confidence": 0.3,
                "tags": [], "anomaly": False, "notes": str(exc)}


# =============================================================================
# Stage 4b — Geospatial Back-Projection
# =============================================================================

def bbox_to_geo(
    bbox: list[float],
    frame_shape: tuple[int, int],
    klv: KLVFrame,
) -> tuple[float, float, float]:
    """
    Back-project a pixel bounding box centre to approximate WGS-84 coordinates
    using the MISB frame-centre + HFOV/VFOV parameters.

    This is a simple linear interpolation — sufficient for the POC.
    A production system should use the full MISB corner-point or range-bearing
    calculation (SMPTE ST 0601 tags 26-29).
    """
    h, w = frame_shape
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    # Fractional position from frame centre (-0.5 to +0.5)
    fx = (cx / w) - 0.5
    fy = (cy / h) - 0.5

    # Approximate ground coverage at slant range
    if klv.slant_range > 0:
        ground_h = klv.slant_range * math.tan(math.radians(klv.hfov / 2)) * 2
        ground_v = klv.slant_range * math.tan(math.radians(klv.vfov / 2)) * 2
    else:
        ground_h = ground_v = klv.sensor_alt * 0.5  # rough fallback

    # Degrees per metre at this latitude
    lat_per_m = 1.0 / 111_320.0
    lon_per_m = 1.0 / (111_320.0 * max(math.cos(math.radians(klv.frame_center_lat)), 0.001))

    lat = klv.frame_center_lat + fy * ground_v * lat_per_m
    lon = klv.frame_center_lon + fx * ground_h * lon_per_m
    alt = klv.sensor_alt  # object altitude unknown; use sensor altitude as proxy

    return lat, lon, alt


# =============================================================================
# Stage 5 — Output Writers
# =============================================================================

class AnnotatedVideoWriter:
    def __init__(self, output_path: str | Path, fps: float, frame_size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    def write(self, frame: np.ndarray, detections: list[Detection]):
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox_xyxy]
            color = (0, 255, 0) if d.frontier_verified else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"[{d.track_id}] {d.label} {d.confidence:.2f}"
            if d.frontier_tags:
                label += f" | {', '.join(d.frontier_tags[:2])}"
            cv2.putText(out, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        self._writer.write(out)

    def release(self):
        self._writer.release()


class GeoJSONWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self._features: list[geojson.Feature] = []

    def add(self, detection: Detection):
        feat = geojson.Feature(
            geometry=geojson.Point((detection.geo_lon, detection.geo_lat, detection.geo_alt)),
            properties={
                "track_id": detection.track_id,
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "frontier_verified": detection.frontier_verified,
                "tags": detection.frontier_tags,
                "pts_us": detection.pts,
            },
        )
        self._features.append(feat)

    def flush(self):
        fc = geojson.FeatureCollection(self._features)
        self.output_path.write_text(geojson.dumps(fc, indent=2))


class AlertEmitter:
    """
    POC: prints JSON alerts to stdout.
    Cloud: swap for aiohttp POST, Kafka producer, or Redis XADD.
    """

    def emit(self, detection: Detection):
        alert = {
            "event": "TARGET_CONFIRMED",
            "track_id": detection.track_id,
            "label": detection.label,
            "confidence": round(detection.confidence, 3),
            "lat": detection.geo_lat,
            "lon": detection.geo_lon,
            "alt": detection.geo_alt,
            "tags": detection.frontier_tags,
            "pts_us": detection.pts,
            "ts_unix": time.time(),
        }
        print(f"[ALERT] {json.dumps(alert)}")


# =============================================================================
# Stage 6 — MVS Trigger (secondary, background stub)
# =============================================================================

class MVSTrigger:
    """
    Accumulates keyframes for tracks that qualify for 3D reconstruction.
    When a track reaches MIN_KEYFRAMES with sufficient parallax, it submits
    the frame bundle to a background COLMAP worker.

    POC: writes frames to disk.  Cloud: enqueue to an async COLMAP job queue.
    """

    MIN_KEYFRAMES = 15
    MIN_PARALLAX_DEG = 3.0  # minimum gimbal angle delta across collected frames

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._keyframes: dict[int, list[tuple[np.ndarray, KLVFrame]]] = {}

    def feed(self, track_id: int, crop: np.ndarray, klv: KLVFrame):
        self._keyframes.setdefault(track_id, []).append((crop, klv))
        frames = self._keyframes[track_id]
        if len(frames) >= self.MIN_KEYFRAMES and self._has_parallax(frames):
            self._submit(track_id, frames)
            del self._keyframes[track_id]

    def _has_parallax(self, frames: list[tuple[np.ndarray, KLVFrame]]) -> bool:
        azimuths = [k.gimbal_az for _, k in frames]
        return (max(azimuths) - min(azimuths)) >= self.MIN_PARALLAX_DEG

    def _submit(self, track_id: int, frames: list[tuple[np.ndarray, KLVFrame]]):
        track_dir = self.output_dir / f"mvs_track_{track_id:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for i, (img, _klv) in enumerate(frames):
            cv2.imwrite(str(track_dir / f"frame_{i:04d}.jpg"), img)
        print(f"[MVS] Track {track_id}: {len(frames)} keyframes queued → {track_dir}")
        # TODO: subprocess.Popen(["colmap", "automatic_reconstructor", "--image_path", str(track_dir), ...])


# =============================================================================
# Main Pipeline Orchestrator
# =============================================================================

def run_pipeline(
    source: str,
    output_dir: str = "pipeline_output",
    model_path: str = "yolo11n.pt",
    sample_fps: float = 4.0,
    use_tiling: bool = False,
    frontier_verify: bool = True,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ingestor = MISBIngestor(source, sample_fps=sample_fps)
    detector = YOLODetector(model_path=model_path, use_tiling=use_tiling)
    tracker = SimpleTracker()
    mvs_trigger = MVSTrigger(out / "mvs")
    geo_writer = GeoJSONWriter(out / "detections.geojson")
    alert_emitter = AlertEmitter()

    # Determine output video dimensions from first frame
    cap_probe = cv2.VideoCapture(source)
    frame_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()
    video_writer = AnnotatedVideoWriter(out / "annotated.mp4", sample_fps, (frame_w, frame_h))

    verification_counts: dict[int, int] = {}

    try:
        for frame, klv in ingestor:
            raw_dets = detector.detect(frame)
            tracked = tracker.update(raw_dets)

            confirmed: list[Detection] = []
            for det in tracked:
                det.pts = klv.pts
                det.geo_lat, det.geo_lon, det.geo_alt = bbox_to_geo(
                    det.bbox_xyxy, frame.shape[:2], klv
                )

                x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size == 0:
                    continue

                count = verification_counts.get(det.track_id, 0)
                needs_verify = frontier_verify and (count % VERIFICATION_INTERVAL == 0)

                if needs_verify:
                    result = verify_with_frontier(crop)
                    det.frontier_verified = result.get("confirmed", False)
                    det.frontier_tags = result.get("tags", [])
                    if result.get("anomaly"):
                        alert_emitter.emit(det)

                verification_counts[det.track_id] = count + 1

                if det.frontier_verified or not frontier_verify:
                    confirmed.append(det)
                    geo_writer.add(det)
                    mvs_trigger.feed(det.track_id, crop, klv)

            video_writer.write(frame, confirmed)

    finally:
        ingestor.release()
        video_writer.release()
        geo_writer.flush()
        print(f"[DONE] Outputs written to {out.resolve()}")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FMV Intelligence Pipeline (MISB/KLV)")
    parser.add_argument("source", help="Path to MPEG-TS file or RTSP URL")
    parser.add_argument("--output-dir", default="pipeline_output")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model weights")
    parser.add_argument("--fps", type=float, default=4.0, help="Sample rate (frames/sec)")
    parser.add_argument("--tiling", action="store_true", help="Enable YOLT-style frame tiling")
    parser.add_argument("--no-frontier", action="store_true", help="Skip frontier model verification")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        output_dir=args.output_dir,
        model_path=args.model,
        sample_fps=args.fps,
        use_tiling=args.tiling,
        frontier_verify=not args.no_frontier,
    )
