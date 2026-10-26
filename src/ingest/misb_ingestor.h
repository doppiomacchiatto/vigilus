#pragma once
/**
 * MISBIngestor — MPEG-TS / RTSP decode + KLV demux
 *
 * Uses FFmpeg libavcodec with NVDEC hardware acceleration.
 * Frames are decoded directly to GPU (NV12 surface) and placed
 * into a thread-safe queue as cv::cuda::GpuMat.
 *
 * KLV packets are extracted from the MPEG-TS data PID in a
 * parallel thread and stored in a ring buffer keyed by PTS.
 */

#include <string>
#include <atomic>

struct KLVFrame {
    int64_t  pts_us        = 0;
    double   sensor_lat    = 0.0;
    double   sensor_lon    = 0.0;
    double   sensor_alt    = 0.0;
    double   platform_hdg  = 0.0;
    double   gimbal_az     = 0.0;
    double   gimbal_el     = -90.0;
    double   slant_range   = 0.0;
    double   hfov          = 5.0;
    double   vfov          = 3.0;
    double   frame_ctr_lat = 0.0;
    double   frame_ctr_lon = 0.0;
};

class MISBIngestor {
public:
    explicit MISBIngestor(const std::string& source, double sample_fps = 4.0);
    ~MISBIngestor();

    /// Blocking run loop — exits when running is set to false or stream ends.
    void run(std::atomic<bool>& running);

private:
    std::string  source_;
    double       sample_fps_;

    // TODO: AVFormatContext, AVCodecContext, CUDA hw device context
    // TODO: KLV ring buffer (lock-free)
};
