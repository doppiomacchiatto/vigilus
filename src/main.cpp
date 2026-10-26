/**
 * Vigilius — FMV Intelligence Pipeline
 * C++ hot path orchestrator (Phase 2)
 *
 * Launches pipeline stage threads and wires them together via
 * thread-safe queues. Each stage is a producer-consumer pair:
 *
 *   MISBIngestor  →  [FrameQueue]  →  YOLODetector + TileScheduler
 *       →  [DetectionQueue]  →  ByteTracker
 *       →  [TrackQueue]      →  GeoBackProjector
 *       →  [GeoQueue]        →  OutputFan (VideoWriter, GeoJSONWriter, KafkaEmitter)
 *                             →  FrontierBridge (async, Python sidecar)
 *                             →  MVSTrigger (background)
 *
 * TODO: implement stage thread launch and queue wiring.
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>

#include "ingest/misb_ingestor.h"
#include "detect/yolo_trt.h"
#include "detect/tile_scheduler.h"
#include "track/bytetrack.h"
#include "geo/backproject.h"
#include "output/video_writer.h"
#include "output/geojson_writer.h"
#include "output/kafka_emitter.h"
#include "ipc/frontier_bridge.h"

static std::atomic<bool> g_running{true};

void signal_handler(int) { g_running = false; }

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: fmv_pipeline <source> [--tiling] [--no-frontier]\n";
        return 1;
    }

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    const std::string source = argv[1];
    std::cout << "[Vigilius] Starting pipeline on: " << source << "\n";

    // TODO: parse flags, instantiate stage objects, launch threads
    // Example thread launch pattern:
    //   auto ingestor = std::make_unique<MISBIngestor>(source);
    //   std::thread ingest_thread([&]{ ingestor->run(g_running); });

    std::cout << "[Vigilius] C++ hot path stub — implement stage threads.\n";
    return 0;
}
