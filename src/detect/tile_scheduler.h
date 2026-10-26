#pragma once
/// YOLT-style tiled chip extraction using cv::cuda for GPU-accelerated slicing.
/// TODO: implement chip extraction with overlap, offset tracking, and de-duplication NMS.
#include <vector>
struct TileChip { int x_offset, y_offset, width, height; };
class TileScheduler {
public:
    TileScheduler(int tile_size = 640, float overlap = 0.2f);
    std::vector<TileChip> schedule(int frame_w, int frame_h) const;
private:
    int   tile_size_;
    float overlap_;
};
