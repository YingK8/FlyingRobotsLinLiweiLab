cmake_minimum_required(VERSION 3.15...3.30)
project(pmw_pose LANGUAGES CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
# Homebrew's OpenCV and Eigen. The cv2 wheel in the venv is a different OpenCV (5.x) and
# its ABI is not shared: frames cross the boundary as numpy buffers, never as cv::Mat.
list(APPEND CMAKE_PREFIX_PATH /opt/homebrew)
# core/imgproc/calib3d only. NOT videoio: Homebrew's libopencv_videoio links ffmpeg 7's
# libavcodec.61 while ffmpeg 8 is installed, so it does not load -- and the repair would
# be `brew` installing OpenCV 5.0, the version bump theory.md 21.2 records as moving
# fitEllipseDirect and remap. Capture goes straight to AVFoundation instead
# (src/capture.h says why at length).
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d)
find_package(Eigen3 REQUIRED)

nanobind_add_module(pmw_pose STABLE_ABI NB_STATIC
  src/bind.cpp src/evidence.cpp src/conic.cpp src/segment.cpp src/trf.cpp src/refine.cpp src/stereo.cpp
  src/capture_avf.mm)
# ARC, for the .mm only -- it is what makes the AVFoundation objects held as C++ members
# of AvfCamera reference-counted correctly.
set_source_files_properties(src/capture_avf.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_include_directories(pmw_pose PRIVATE ${OpenCV_INCLUDE_DIRS})
# No fused multiply-add: numpy rounds every product and sum separately, and the
# analytic Jacobian's normal columns are a forward difference at sqrt(eps), which
# amplifies a one-ulp difference in the rim points to 1e-5 of the Jacobian
# (theory.md 21.3). Parity with the reference needs the same rounding.
target_compile_options(pmw_pose PRIVATE -ffp-contract=off)
target_link_libraries(pmw_pose PRIVATE ${OpenCV_LIBS} Eigen3::Eigen
  "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo" "-framework Foundation")
# Without the rpath dyld cannot find the Homebrew dylibs at import time.
set_target_properties(pmw_pose PROPERTIES INSTALL_RPATH "/opt/homebrew/lib")
install(TARGETS pmw_pose LIBRARY DESTINATION .)
