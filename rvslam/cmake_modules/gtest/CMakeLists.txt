cmake_minimum_required(VERSION 2.8.7)
project(gtest_builder C CXX)

# Refer to http://www.kitware.com/media/html/BuildingExternalProjectsWithCMake2.8.html
# This is cmake internal include statement
include(ExternalProject)

## Failed attempt to stop downloading gtest when it exists..
## the if stmt is executed when cmake is running, not make is running.
## Comment/uncomment the UPDATE_COMMAND and CONFIGURE_COMMAND.

#if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}-build")
#message("-- downloaded gtest found (${CMAKE_CURRENT_BINARY_DIR}-build)")

ExternalProject_Add(googletest
     SVN_REPOSITORY http://googletest.googlecode.com/svn/trunk
   TIMEOUT 5
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_FLAGS=${GTEST_CXX_FLAGS}
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
#    UPDATE_COMMAND ""  # Disable update.
#    CONFIGURE_COMMAND "" # Disable configure also.
#    BUILD_COMMAND "" # Disable build also.
    INSTALL_COMMAND "" # Disable install step.
)

# Specify include dir
ExternalProject_Get_Property(googletest source_dir)
set(GTEST_INCLUDE_DIRS ${source_dir}/include PARENT_SCOPE)
message("-- GTEST_INCLUDE_DIRS: ${GTEST_INCLUDE_DIRS}")

# Specify MainTest's link libraries
ExternalProject_Get_Property(googletest binary_dir)
set(GTEST_LIBS_DIR ${binary_dir} PARENT_SCOPE)
message("-- GTEST_LIBS_DIR: ${GTEST_LIBS_DIR}")
# Actual libraries.
#set(GTEST_LIBS ${GTEST_LIBS_DIR}/libgtest.a ${GTEST_LIBS_DIR}/libgtest_main.a)

