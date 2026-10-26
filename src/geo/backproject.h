#pragma once
/// CRS back-projection — pixel bbox → WGS-84 coordinates.
/// FMV mode: uses KLV frame-center + FOV interpolation.
/// Raster mode: uses GDAL geotransform (exact).
#include "../ingest/misb_ingestor.h"
struct GeoPoint { double lat, lon, alt; };
class BackProjector {
public:
    /// KLV-based back-projection (FMV)
    static GeoPoint from_klv(float cx, float cy, int frame_w, int frame_h, const KLVFrame& klv);
    /// GDAL geotransform back-projection (raster) — TODO Task #4
    static GeoPoint from_geotransform(float cx, float cy, const double gt[6]);
};
