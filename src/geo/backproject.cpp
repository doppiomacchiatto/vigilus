/// BackProjector stub — TODO: wire PROJ/GDAL C API.
#include "backproject.h"
#include <cmath>
GeoPoint BackProjector::from_klv(float cx, float cy, int w, int h, const KLVFrame& klv) {
    double fx = (cx / w) - 0.5, fy = (cy / h) - 0.5;
    double ground_h = klv.slant_range > 0
        ? klv.slant_range * std::tan(klv.hfov / 2.0 * M_PI / 180.0) * 2.0
        : klv.sensor_alt * 0.5;
    double ground_v = klv.slant_range > 0
        ? klv.slant_range * std::tan(klv.vfov / 2.0 * M_PI / 180.0) * 2.0
        : klv.sensor_alt * 0.5;
    double lat_per_m = 1.0 / 111320.0;
    double lon_per_m = 1.0 / (111320.0 * std::max(std::cos(klv.frame_ctr_lat * M_PI / 180.0), 0.001));
    return { klv.frame_ctr_lat + fy * ground_v * lat_per_m,
             klv.frame_ctr_lon + fx * ground_h * lon_per_m,
             klv.sensor_alt };
}
GeoPoint BackProjector::from_geotransform(float cx, float cy, const double gt[6]) {
    // gt: GDAL affine geotransform [originX, pixelW, rotX, originY, rotY, pixelH]
    double lon = gt[0] + cx * gt[1] + cy * gt[2];
    double lat = gt[3] + cx * gt[4] + cy * gt[5];
    return { lat, lon, 0.0 };
}
