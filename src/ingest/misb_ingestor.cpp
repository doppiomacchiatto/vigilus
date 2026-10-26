/**
 * MISBIngestor implementation stub.
 * TODO Phase 2:
 *   1. Open source with avformat_open_input() + CUDA hw device context
 *   2. Find video stream + data (KLV) stream PIDs
 *   3. Decode video frames via NVDEC → cv::cuda::GpuMat (NV12, zero-copy)
 *   4. Parse KLV data packets with libklv into KLVFrame ring buffer
 *   5. Push (GpuMat, KLVFrame) pairs onto downstream FrameQueue
 */

#include "misb_ingestor.h"
#include <iostream>

MISBIngestor::MISBIngestor(const std::string& source, double sample_fps)
    : source_(source), sample_fps_(sample_fps) {}

MISBIngestor::~MISBIngestor() {}

void MISBIngestor::run(std::atomic<bool>& running) {
    std::cout << "[MISBIngestor] stub — source: " << source_ << "\n";
    // TODO: implement FFmpeg + NVDEC decode loop
    (void)running;
}
