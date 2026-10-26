#pragma once
/// ByteTrack C++ — IoU-based temporal tracker.
/// TODO: implement full ByteTrack with Kalman filter state estimation.
#include <vector>
#include "../detect/yolo_trt.h"
class ByteTracker {
public:
    ByteTracker(float iou_threshold = 0.4f, int max_age = 10);
    std::vector<Detection> update(std::vector<Detection>& detections);
private:
    float iou_threshold_;
    int   max_age_;
    int   next_id_ = 1;
    // TODO: active track state (Kalman filters, age counters)
};
