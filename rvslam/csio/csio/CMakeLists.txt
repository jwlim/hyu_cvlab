# project
cmake_minimum_required(VERSION 2.6)
project(csio)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake_modules/")

# make release version
set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(CMAKE_BUILD_TYPE release)
set(CMAKE_VERBOSE_MAKEFILE yes)

enable_testing()

# Default locations to search for on various platforms.
LIST(APPEND SEARCH_LIBS /usr/lib)
LIST(APPEND SEARCH_LIBS /usr/local/lib)
LIST(APPEND SEARCH_LIBS /opt/local/lib)

LIST(APPEND SEARCH_HEADERS /usr/include)
LIST(APPEND SEARCH_HEADERS /usr/local/include)
LIST(APPEND SEARCH_HEADERS /opt/local/include)

# Google libraries
message("-- Check for Google Log")
find_library(GLOG_LIB NAMES glog PATHS ${SEARCH_LIBS})
find_path(GLOG_INCLUDE NAMES glog/logging.h PATHS ${SEARCH_HEADERS})

message("-- Check for Google Flags")
find_library(GFLAGS_LIB NAMES gflags PATHS ${SEARCH_LIBS})
find_path(GFLAGS_INCLUDE NAMES gflags/gflags.h PATHS ${SEARCH_HEADERS})


message("-- Check for Google Test")
find_library(GTEST_LIB NAMES gtest PATHS ${SEARCH_LIBS})
find_path(GTEST_INCLUDE NAMES gtest/gtest.h PATHS ${SEARCH_HEADERS})

set(GOOGLE_LIBRARIES ${GFLAGS_LIB} ${GLOG_LIB} ${GTEST_LIB})

# Other libraries
MESSAGE("-- Check for PNG ")
find_package(PNG)

message("-- Check for libfreenect (optional)")
find_library(FREENECT_LIB NAMES freenect PATHS ${SEARCH_LIBS})
find_path(FREENECT_INCLUDE NAMES libfreenect/libfreenect.h
          PATHS ${SEARCH_HEADERS})

message("-- Check for GLUT (optional)")
if (APPLE)
  add_definitions(-DAPPLE)
  set(CMAKE_EXE_LINKER_FLAGS "-framework OpenGL -framework GLUT")
else (APPLE)  # Linux, not so much
  find_package(OpenGL REQUIRED)
  find_package(GLUT REQUIRED)
  include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
endif (APPLE)

# directories
set(CSIO_INCLUDE_DIR .)

# include directory
include_directories(${CSIO_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE})
include_directories(${GFLAGS_INCLUDE})
include_directories(${GTEST_INCLUDE})

# use sse3 instruction set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -msse3")  # -DNDEBUG

add_executable(csio_test src/csio_test.cc)
target_link_libraries(csio_test ${GOOGLE_LIBRARIES} pthread)
add_test(csio_test csio_test)

#if (FREENECT_LIB)
#  include_directories(${FREENECT_INCLUDE})
#  add_executable(csiomod_freenect src/csiomod_freenect.cc)
#  target_link_libraries(csiomod_freenect ${FREENECT_LIB} ${GOOGLE_LIBRARIES} 
#                        pthread)
#endif (FREENECT_LIB)

add_executable(csio_glviewer
    src/csio_glviewer.cc src/csio_glviewer_view.cc src/image_file.cc)
target_link_libraries(csio_glviewer ${GOOGLE_LIBRARIES} ${PNG_LIBRARY} pthread
                      ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(spm2csio src/spm2csio.cc)
target_link_libraries(spm2csio ${GOOGLE_LIBRARIES} ${PNG_LIBRARY} pthread)

