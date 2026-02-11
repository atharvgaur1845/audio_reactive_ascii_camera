cmake_minimum_required(VERSION 3.14)
project(ascii_cam CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio objdetect video)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

add_executable(ascii_cam ascii_cam.cpp main.cpp)

target_include_directories(ascii_cam PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PORTAUDIO_INCLUDE_DIRS}
)

target_link_libraries(ascii_cam PRIVATE
    ${OpenCV_LIBS}
    ${PORTAUDIO_LIBRARIES}
    pthread
)
