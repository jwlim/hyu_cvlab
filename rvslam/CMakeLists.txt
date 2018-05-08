cmake_minimum_required(VERSION 2.6)
project(rvslam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake_modules/")

# make release version
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_VERBOSE_MAKEFILE yes)
set(CMAKE_CXX_CFLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS "-g -O3 -std=c++11 -msse3 -DNDEBUG")

enable_testing()

if (APPLE)
  add_definitions(-DAPPLE)
endif (APPLE)

find_package(Eigen3 3.1.3 REQUIRED)
find_package(GoogleLibs REQUIRED)
find_package(PNG REQUIRED STATIC)
find_package(JPEG REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV)  # Optional : for features only.

#Camera
LIST(APPEND SEARCH_HEADERS /usr/local/include/libfreenect)
LIST(APPEND SEARCH_HEADERS /usr/local/include/dc1394)
message("-- Check for libfreenect (optional)")
find_library(FREENECT_LIB NAMES freenect PATHS ${SEARCH_LIBS})
find_path(FREENECT_INCLUDE NAMES libfreenect.hpp PATHS ${SEARCH_HEADERS})

message("-- Check for libdc1394 (optional)")
find_library(DC1394_LIB NAMES dc1394 PATHS ${SEARCH_LIBS})
find_path(DC1394_INCLUDE NAMES dc1394.h PATHS ${SEARCH_HEADERS})

if (APPLE)
  message("libusb added for libfreenect")
  include_directories(/usr/local/include/libusb-1.0)
else (APPLE)
  message("libusb added for libfreenect")
  include_directories(/usr/include/libusb-1.0)
endif (APPLE)
set(SRC_DIR src)

include_directories(${SRC_DIR})
include_directories(${GLOG_INCLUDE})
include_directories(${GFLAGS_INCLUDE})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(.)

#----------------------------------------
# Third party libraries
#----------------------------------------

#ADD_SUBDIRECTORY(third_party)

#----------------------------------------
# rvslam Libraries
#----------------------------------------

if (OpenCV_FOUND)
  set(RVSLAM_OPENCV_SRC
      src/feat_opencv.cc #src/feat_harris_klt.cc
      src/loopclosing.cc
  )
endif (OpenCV_FOUND)

## rvslam library
add_library(rvslam
    src/image_file.cc src/image_util.cc
    src/klt_tracker.cc src/p3p.cc src/five_point.cc src/triangulation.cc
    src/homography.cc src/estimate_rot.cc src/visual_odometer.cc src/ransac.cc 
    src/similarity.cc src/rvmap.cc src/optimize.cc src/sim3solver.cc src/fastdetector.cc
    src/featuretracker.cc
    csio/src/csiomod_camera1394.cc
    
    ${RVSLAM_OPENCV_SRC}
)
target_link_libraries(rvslam
    ${PNG_LIBRARY} ${CERES_LIBRARIES} ${GOOGLE_LIBRARIES} pthread
    ${OpenCV_LIBS}  # Optional libraries.
    ${DC1394_LIB} -lpthread
)


#----------------------------------------
# rvslam Executables
#----------------------------------------

# KLT tracker
add_executable(klt_tracker src/klt_tracker_main.cc)
target_link_libraries(klt_tracker rvslam)

#if (OpenCV_FOUND)
  add_executable(rvslam_main src/rvslam_main.cc)
  target_link_libraries(rvslam_main rvslam)
#endif (OpenCV_FOUND)


