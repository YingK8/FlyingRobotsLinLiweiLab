cmake_minimum_required(VERSION 3.15...3.30)
project(pmw_pose LANGUAGES CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

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

# ---------------------------------------------------------------------------------------
# pmw_core: every numeric translation unit, built ONCE and shared by both front-ends.
#
# The split exists so the standalone controller (`pmw_fly`) and the Python module
# (`pmw_pose`) cannot drift into different arithmetic. `native_parity.py` holds `pmw_pose`
# to the Python reference; if the binary were compiled from its own copy of these sources
# with its own flags, parity could pass while the thing that actually flies computed
# something else. One object library, one set of flags, both front-ends linked against it.
#
# `-ffp-contract=off` therefore belongs HERE and nowhere else. See the note below.
# ---------------------------------------------------------------------------------------
add_library(pmw_core OBJECT
  src/evidence.cpp src/conic.cpp src/segment.cpp src/trf.cpp src/refine.cpp src/stereo.cpp
  src/capture_avf.mm)
set_target_properties(pmw_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pmw_core PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pmw_core PUBLIC ${OpenCV_LIBS} Eigen3::Eigen)
# ARC, for the .mm only -- it is what makes the AVFoundation objects held as C++ members
# of AvfCamera reference-counted correctly.
set_source_files_properties(src/capture_avf.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
# No fused multiply-add: numpy rounds every product and sum separately, and the
# analytic Jacobian's normal columns are a forward difference at sqrt(eps), which
# amplifies a one-ulp difference in the rim points to 1e-5 of the Jacobian
# (theory.md 21.3). Parity with the reference needs the same rounding.
target_compile_options(pmw_core PUBLIC -ffp-contract=off)

set(PMW_FRAMEWORKS "-framework AVFoundation" "-framework CoreMedia" "-framework CoreVideo"
                   "-framework Foundation")

# ---------------------------------------------------------------------------------------
# pmw_pose: the Python module. src/bind.cpp is the only file that touches Python.
# ---------------------------------------------------------------------------------------
# ON by default so `uv sync --extra native` and the wheel are unchanged. OFF lets the
# standalone binary build on a machine with no nanobind -- and, more to the point, proves
# the binary genuinely has no Python dependency rather than merely claiming one.
option(PMW_MODULE "build the pmw_pose Python module" ON)
if(PMW_MODULE)
  find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(pmw_pose STABLE_ABI NB_STATIC src/bind.cpp)
  target_link_libraries(pmw_pose PRIVATE pmw_core ${PMW_FRAMEWORKS})
  # Without the rpath dyld cannot find the Homebrew dylibs at import time.
  set_target_properties(pmw_pose PROPERTIES INSTALL_RPATH "/opt/homebrew/lib")
  install(TARGETS pmw_pose LIBRARY DESTINATION .)
endif()

# ---------------------------------------------------------------------------------------
# pmw_fly: the standalone controller. OFF by default, so `uv sync --extra native` and the
# wheel it builds are byte-identical to what they were before this option existed -- the
# binary is opted into explicitly:
#
#   cmake -S controller/native -B build/fly -DPMW_FLY=ON && cmake --build build/fly
#
# No Python, no nanobind: src/fly.cpp is the only file with a main().
# ---------------------------------------------------------------------------------------
option(PMW_FLY "build the standalone controller binary" OFF)
if(PMW_FLY)
  add_executable(pmw_fly src/fly.cpp src/link.cpp)
  target_include_directories(pmw_fly PRIVATE ../../lib/DriveFrame/src)
  target_link_libraries(pmw_fly PRIVATE pmw_core ${PMW_FRAMEWORKS})
  set_target_properties(pmw_fly PROPERTIES INSTALL_RPATH "/opt/homebrew/lib"
                                           BUILD_WITH_INSTALL_RPATH ON)
endif()
