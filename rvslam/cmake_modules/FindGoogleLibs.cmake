# FindGoogleLibs.cmake
#
# Mostly from ceres-solver/CMakeLists.txt

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

#message("-- Add dependencies for Google Test. You need internet connection.")
#add_subdirectory(${CMAKE_MODULE_PATH}/gtest)
#set(GTEST_LIBS ${GTEST_LIBS_DIR}/libgtest.a ${GTEST_LIBS_DIR}/libgtest_main.a)

set(GOOGLE_LIBRARIES ${GFLAGS_LIB} ${GLOG_LIB})
set(GOOGLE_TEST_LIBRARIES ${GFLAGS_LIB} ${GLOG_LIB})
#set(GOOGLE_TEST_LIBRARIES ${GFLAGS_LIB} ${GLOG_LIB} ${GTEST_LIBS})
