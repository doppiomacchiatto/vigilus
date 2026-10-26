#pragma once
/**
 * YoloTRT — TensorRT YOLO engine wrapper
 *
 * Owns the TensorRT IRuntime, ICudaEngine, and IExecutionContext
 * via std::unique_ptr with custom deleters (RAII).
 *
 * Input/output bindings are pre-allocated cudaMalloc buffers so
 * inference never touches host memory — the entire path from
 * cv::cuda::GpuMat resize → TensorRT → detection tensors stays in VRAM.
 *
 * Usage:
 *   YoloTRT detector("yolo11n.engine");
 *   auto detections = detector.infer(gpu_mat);  // returns host-side results
 */

#include <string>
#include <vector>
#include <memory>

struct Detection {
    int     track_id    = -1;
    int     class_id    = 0;
    float   confidence  = 0.0f;
    float   x1, y1, x2, y2;   // pixel coords in original frame
    double  geo_lat     = 0.0;
    double  geo_lon     = 0.0;
    double  geo_alt     = 0.0;
    bool    frontier_verified = false;
};

class YoloTRT {
public:
    explicit YoloTRT(const std::string& engine_path, float conf_threshold = 0.35f);
    ~YoloTRT();

    /// Run inference on a GPU image buffer. Returns detections on host.
    std::vector<Detection> infer(/* cv::cuda::GpuMat& frame */);

private:
    std::string engine_path_;
    float       conf_threshold_;

    // TensorRT objects — held via unique_ptr with custom deleters
    // TODO: IRuntime*, ICudaEngine*, IExecutionContext* with TRTDeleter
    // TODO: pre-allocated CUDA input/output binding buffers
};
