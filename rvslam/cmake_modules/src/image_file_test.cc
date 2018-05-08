#include <iostream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "image_file.h"

using namespace std;
using namespace rvslam;

namespace {

TEST(ImageFileTest, TestReadImageGray8) {
/*
  ImageGray8 gray;
  EXPECT_TRUE(ReadImage("test_gray8.png", &gray));
  LOG(INFO) << "gray " << gray.rows() << "x" << gray.cols();
  EXPECT_TRUE(WriteImageGray8(gray, "test_gray8_out.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &gray));
  EXPECT_TRUE(WriteImageGray8(gray, "test_rgb_gray8_out.png"));
*/
  MCImageGray8 mc_gray;
  EXPECT_TRUE(ReadImage("test_gray8.png", &mc_gray));
  LOG(INFO) << "mc_gray " << mc_gray.rows() << "x" << mc_gray.cols();
  EXPECT_TRUE(WriteImageGray8(mc_gray, "test_mc_gray8_out.png"));
  EXPECT_TRUE(ReadImage("test_rgb8.png", &mc_gray));
  EXPECT_TRUE(WriteImageGray8(mc_gray, "test_rgb_mc_gray8_out.png"));
}

TEST(ImageFileTest, TestReadImageRGB8) {
  MCImageRGB8 rgb;
  EXPECT_TRUE(ReadImageRGB8("test_rgb8.png", &rgb));
  LOG(INFO) << "rgb " << rgb.rows() << "x" << rgb.cols();
  EXPECT_TRUE(WriteImageRGB8(rgb, "test_rgb8_out.png"));
  EXPECT_TRUE(ReadImageRGB8("test_gray8.png", &rgb));
  LOG(INFO) << "rgb " << rgb.rows() << "x" << rgb.cols();
  EXPECT_TRUE(WriteImageRGB8(rgb, "test_gray_rgb8_out.png"));
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
