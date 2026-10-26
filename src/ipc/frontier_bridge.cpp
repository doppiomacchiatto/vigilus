/// FrontierBridge stub — TODO: implement Unix socket connect + async send/recv loop.
#include "frontier_bridge.h"
#include <iostream>
FrontierBridge::FrontierBridge(const std::string& socket_path) : socket_path_(socket_path) {}
void FrontierBridge::connect() {
    std::cout << "[FrontierBridge] TODO: connect to " << socket_path_ << "\n";
    // TODO: socket(AF_UNIX, SOCK_STREAM, 0) + connect()
}
void FrontierBridge::disconnect() { if (sock_fd_ >= 0) { /* close(sock_fd_) */ sock_fd_ = -1; } }
void FrontierBridge::submit_async(int track_id, const std::string&, FrontierCallback cb) {
    // TODO: serialize + send over socket; recv response on callback thread
    std::cout << "[FrontierBridge] stub submit track_id=" << track_id << "\n";
    cb(track_id, true, "unknown", 0.5f);
}
