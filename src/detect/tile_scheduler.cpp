/// TileScheduler stub — TODO: implement YOLT chip scheduling and cross-tile NMS.
#include "tile_scheduler.h"
TileScheduler::TileScheduler(int tile_size, float overlap)
    : tile_size_(tile_size), overlap_(overlap) {}
std::vector<TileChip> TileScheduler::schedule(int frame_w, int frame_h) const {
    std::vector<TileChip> chips;
    int stride = static_cast<int>(tile_size_ * (1.0f - overlap_));
    for (int y = 0; y < frame_h; y += stride)
        for (int x = 0; x < frame_w; x += stride)
            chips.push_back({x, y,
                std::min(tile_size_, frame_w - x),
                std::min(tile_size_, frame_h - y)});
    return chips;
}
