// csio_image_test.cc
//

#include <fstream>
#include <sstream>

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "image.h"  // csio/image.h

using namespace std;

TEST(CSIO_IMAGE, CSIOTest) {
}

TEST(CSIO_IMAGE, CSIOInputOutputStreamTest) {
}

TEST(CSIO_IMAGE, CSIOEscapeStrTest) {
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}

