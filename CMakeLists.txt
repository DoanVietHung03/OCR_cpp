cmake_minimum_required(VERSION 3.21)
project(OCR_cpp VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(OpenCV_DIR "D:/OpenCV_CUDA/build")
find_package(OpenCV REQUIRED)

# Khai báo đường dẫn đến thư mục ONNX Runtime
set(ONNXRUNTIME_DIR "D:/onnxruntime-gpu/onnxruntime-gpu")
# Thêm thư mục include để nhận diện #include <onnxruntime_cxx_api.h>
include_directories(${ONNXRUNTIME_DIR}/include)

add_executable(build_app 
    "src/main.cpp"
    "src/BYTETracker.cpp" 
    "src/kalmanFilter.cpp" 
    "src/STrack.cpp" 
    "src/lapjv.cpp"
    "src/utils.cpp"
)

target_include_directories(build_app PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(build_app PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(build_app PRIVATE "C:/Users/ACER/OneDrive/Desktop/OCR_cpp/eigen-3.4.0")

find_package(Threads REQUIRED)

target_link_libraries(build_app PRIVATE 
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib 
    Threads::Threads)

add_custom_command(TARGET build_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:build_app>
    $<TARGET_FILE_DIR:build_app>
    COMMAND_EXPAND_LISTS
    COMMENT "Dang copy tu dong cac file DLL can thiet vao thu muc chay..."
)

add_custom_command(TARGET build_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/weights"
    "$<TARGET_FILE_DIR:build_app>/weights"
    COMMENT "Dang copy tu dong thu muc weights vao thu muc chay..."
)

add_custom_command(TARGET build_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/CCTV"
    "$<TARGET_FILE_DIR:build_app>/CCTV"
    COMMENT "Dang copy tu dong thu muc CCTV vao thu muc chay..."
)

add_custom_command(TARGET build_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${ONNXRUNTIME_DIR}/lib"
    "$<TARGET_FILE_DIR:build_app>"
    COMMENT "Dang copy file dll cua ONNX Runtime vao thu muc chay..."
)