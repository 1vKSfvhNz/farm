// backend/cpp_src/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(farm_manager_cpp)

# Configuration C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -pthread")

# Trouver pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Options CUDA (optionnel)
option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
endif()

# Options OpenCV
find_package(OpenCV REQUIRED)

# Fichiers sources
set(SOURCES
    src/growth_predictor.cpp
    src/video_analyzer.cpp
    src/sensor_gateway.cpp
    src/tensorrt_yolo.cpp
    src/botsort_tracker.cpp
)

# En-têtes
set(HEADERS
    include/growth_predictor.hpp
    include/video_analyzer.hpp
    include/sensor_gateway.hpp
    include/tensorrt_yolo.hpp
    include/botsort_tracker.hpp
)

# Bibliothèque principale
pybind11_add_module(growth_cpp src/growth_predictor.cpp)
pybind11_add_module(video_detector_cpp src/video_analyzer.cpp src/tensorrt_yolo.cpp src/botsort_tracker.cpp)
pybind11_add_module(sensor_cpp src/sensor_gateway.cpp)

# Liens
target_link_libraries(growth_cpp PRIVATE ${OpenCV_LIBS})
target_link_libraries(video_detector_cpp PRIVATE ${OpenCV_LIBS})
target_link_libraries(sensor_cpp PRIVATE)

if(USE_CUDA)
    target_link_libraries(video_detector_cpp PRIVATE ${CUDA_LIBRARIES})
endif()

# Installation
install(TARGETS growth_cpp video_detector_cpp sensor_cpp DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)