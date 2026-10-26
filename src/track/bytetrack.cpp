/// ByteTracker stub — TODO: implement Kalman filter + IoU assignment.
#include "bytetrack.h"
ByteTracker::ByteTracker(float iou_threshold, int max_age)
    : iou_threshold_(iou_threshold), max_age_(max_age) {}
std::vector<Detection> ByteTracker::update(std::vector<Detection>& detections) {
    // TODO: match detections to existing tracks via IoU, assign IDs, age out lost tracks
    for (auto& d : detections) d.track_id = next_id_++;
    return detections;
}
