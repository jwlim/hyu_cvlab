#include <iostream>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "image_pyramid.h"
#include "image.h"

using namespace std;
using namespace rvslam;

namespace {

typedef Eigen::ArrayXXf ArrayF;

TEST(ImagePyramidTest, TestImagePyramid) {
  // Make a simple checkerboard pattern.
  ArrayF img(48, 32);
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      img(x, y) = img(x + 16, y + 16) = img(x + 32, y) = 1.0;
    }
  }
  LOG(INFO) << setfill(' ') << endl << img;

  ImagePyramidBuilder pyramid_builder(3, 0.5, 3);
  ImagePyramid pyramid;
  pyramid_builder.Build(img, &pyramid);
  LOG(INFO) << setfill(' ') << endl << pyramid_builder.gaussian_1d_kernel();
  for (int i = 0; i < pyramid.num_levels(); ++i) {
    const ImagePyramid::Level& lvl = pyramid[i];
    LOG(INFO) << setfill(' ') << endl << lvl.imgf;
    LOG(INFO) << setfill(' ') << setw(5) << endl << lvl.imgx;
  }
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
