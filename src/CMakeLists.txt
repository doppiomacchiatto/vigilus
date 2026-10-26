cmake_minimum_required(VERSION 3.22)
project(fmv_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Find dependencies ─────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GDAL REQUIRED)

# TensorRT — set TENSORRT_ROOT in cmake invocation or environment
set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation root")
find_library(TENSORRT_LIB nvinfer HINTS ${TENSORRT_ROOT}/lib REQUIRED)
find_library(TENSORRT_ONNX_LIB nvonnxparser HINTS ${TENSORRT_ROOT}/lib)

# Kafka
find_library(RDKAFKA_LIB rdkafka++ REQUIRED)

# ── Source files ──────────────────────────────────────────────────────────────
set(SOURCES
    main.cpp
    ingest/misb_ingestor.cpp
    detect/yolo_trt.cpp
    detect/tile_scheduler.cpp
    track/bytetrack.cpp
    geo/backproject.cpp
    output/video_writer.cpp
    output/geojson_writer.cpp
    output/kafka_emitter.cpp
    ipc/frontier_bridge.cpp
)

add_executable(fmv_pipeline ${SOURCES})

# ── Include directories ───────────────────────────────────────────────────────
target_include_directories(fmv_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${GDAL_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${CUDA_INCLUDE_DIRS}
)

# ── Link libraries ────────────────────────────────────────────────────────────
target_link_libraries(fmv_pipeline PRIVATE
    ${OpenCV_LIBS}
    ${GDAL_LIBRARIES}
    ${TENSORRT_LIB}
    ${TENSORRT_ONNX_LIB}
    ${RDKAFKA_LIB}
    ${CUDA_LIBRARIES}
    avcodec avformat avutil swscale   # FFmpeg
    pthread
)

# ── Compiler flags ────────────────────────────────────────────────────────────
target_compile_options(fmv_pipeline PRIVATE
    -O3 -Wall -Wextra -Wpedantic
    $<$<CONFIG:Debug>:-g -fsanitize=address>
)
