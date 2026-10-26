#pragma once
/**
 * FrontierBridge — Unix domain socket IPC to Python frontier model sidecar.
 *
 * The C++ hot path sends crop JPEG + track metadata over a Unix socket.
 * The Python sidecar calls the frontier model API and returns structured JSON.
 * This keeps the network-bound LLM call in Python without blocking the C++ thread.
 *
 * Message format (length-prefixed MessagePack):
 *   { "track_id": int, "jpeg_b64": str, "klv": { ... } }
 * Response:
 *   { "confirmed": bool, "class": str, "confidence": float, "tags": [...], "anomaly": bool }
 */
#include <string>
#include <functional>
#include "../detect/yolo_trt.h"

using FrontierCallback = std::function<void(int track_id, bool confirmed,
                                             const std::string& cls, float conf)>;
class FrontierBridge {
public:
    explicit FrontierBridge(const std::string& socket_path = "/tmp/vigilius_frontier.sock");
    void submit_async(int track_id, const std::string& jpeg_b64, FrontierCallback cb);
    void connect();
    void disconnect();
private:
    std::string socket_path_;
    int         sock_fd_ = -1;
};
