// image_test.cc

// Define this symbol to enable runtime tests for allocations
#define EIGEN_RUNTIME_NO_MALLOC

#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "image.h"

using namespace std;
using namespace rvslam;

namespace {

#define DUMP_IMAGE(str, img) \
  LOG(INFO) << setfill(' ') << str << ": " \
      << img.width() << "x" << img.height() << "x" << img.depth() << endl \
      << img

#define DUMP_ARRAY(str, array) \
  LOG(INFO) << setfill(' ') << str << ": " \
      << array.rows() << "x" << array.cols() << endl << array

TEST(ImageTest, TestImage) {
  Image<uint8_t> img_gray;
  Image<float> img_float;
  uint8_t test_data[] = { 43, 45, 46, 43, 45, 46, 43, 45, 46, 43, 45, 46 };
  img_gray.FromBuffer(test_data, 4, 3);
  DUMP_IMAGE("img_gray", img_gray);
  img_float = img_gray.cast<float>();
  DUMP_IMAGE("img_float", img_float);
  img_gray.Resize(8, 4);
  img_float.Resize(8, 5);
  img_gray.fill('a');
  img_float.fill('b');
  DUMP_IMAGE("img_gray", img_gray);
  DUMP_IMAGE("img_float", img_float);
}

TEST(MCImageTest, TestMCImage) {
  MCImage<uint8_t, 1> img_gray;
  MCImage<uint8_t, 3> img_rgb;
  uint8_t test_data[] = { 43, 45, 46, 43, 45, 46, 43, 45, 46, 43, 45, 46 };
  img_gray.FromBuffer(test_data, 4, 3, 1);
  img_rgb.FromBuffer(test_data, 2, 2, 3);
  DUMP_IMAGE("img_gray", img_gray);
  DUMP_IMAGE("img_rgb", img_rgb);
  DUMP_ARRAY("img_gray", img_gray.GetPlane());
  DUMP_ARRAY("img_rgb.0", img_rgb.GetPlane(0));
  DUMP_ARRAY("img_rgb.1", img_rgb.GetPlane(1));
Eigen::internal::set_is_malloc_allowed(false);
  img_gray.fill('0');
  img_rgb.fill('1');
Eigen::internal::set_is_malloc_allowed(true);
  DUMP_IMAGE("img_gray", img_gray);
  DUMP_IMAGE("img_rgb", img_rgb);
  img_gray.Resize(8, 4);
  img_rgb.Resize(8, 5);
  img_gray.fill('a');
  img_rgb.fill('b');
  DUMP_IMAGE("img_gray", img_gray);
  DUMP_IMAGE("img_rgb", img_rgb);
  img_gray.Resize(8, 5);
  img_gray.fill('c');
  img_rgb.SetPlane(0, img_gray);
  DUMP_IMAGE("img_rgb", img_rgb);

  MCImage<uint8_t, Eigen::Dynamic> img_test(img_rgb);
  DUMP_IMAGE("img_test", img_test);
  img_test = img_gray;
  DUMP_IMAGE("img_test", img_test);

  MCImage<float, Eigen::Dynamic> img_copy;
  img_copy.Copy(img_gray);
  DUMP_IMAGE("img_copy", img_copy);
  img_copy.Copy(img_rgb);
  DUMP_IMAGE("img_copy", img_copy);
  img_copy = img_rgb.cast<float>();
  DUMP_IMAGE("img_copy", img_copy);

  MCImage<float, 1>::ArrayType array_collapsed;
  img_rgb.Collapse(&array_collapsed);
  DUMP_ARRAY("array_collapsed", array_collapsed.cast<int>());

  MCImageGray8::ArrayType array8 = img_gray.ToArray();
}

TEST(ImageTest, TestFloatImage) {
  ImageGray8 test;
  test.Resize(10, 10);
  test.fill('a');

  ImageFloat tmp = test.cast<float>();
  DUMP_IMAGE("test", test);
  DUMP_IMAGE("tmp", tmp);
}

/*
uint8_t Interp2Ref(const MCImageGray8& image, double x, double y) {
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const int x1 = x0 + 1, y1 = y0 + 1;
  const double rx = x - x0, ry = y - y0;
  LOG(INFO) << "Interp2Ref: " << static_cast<int>(image(x0, y0))
      << ", " << static_cast<int>(image(x1, y0))
      << ", " << static_cast<int>(image(x0, y1))
      << ", " << static_cast<int>(image(x1, y1))
      << " - " << rx << "," << ry << " : "
      << (image(x0, y0) * (1 - rx) + image(x1, y0) * rx) * (1 - ry) +
         (image(x0, y1) * (1 - rx) + image(x1, y1) * rx) * ry;
  return static_cast<uint8_t>(
      (image(x0, y0) * (1 - rx) + image(x1, y0) * rx) * (1 - ry) +
      (image(x0, y1) * (1 - rx) + image(x1, y1) * rx) * ry);
}

TEST(MCImageUtilTest, TestInterp2) {
  MCImageGray8 image8(10, 10);
  for (int r = 0; r < image8.rows(); ++r) {
    for (int c = 0; c < image8.cols(); ++c) {
      image8(r, c) = r + c * image8.rows();
    }
  }
  EXPECT_EQ(Interp2Ref(image8, 0.0, 0.0), Interp2(image8, 0.0, 0.0));
  EXPECT_EQ(Interp2Ref(image8, 5.0, 0.5), Interp2(image8, 5.0, 0.5));
  EXPECT_EQ(Interp2Ref(image8, 0.0, 3.5), Interp2(image8, 0.0, 3.5));
  EXPECT_EQ(Interp2Ref(image8, 3.5, 7.5), Interp2(image8, 3.5, 7.5));
  EXPECT_EQ(Interp2Ref(image8, 5.2, 4.5), Interp2(image8, 5.2, 4.5));
  EXPECT_EQ(Interp2Ref(image8, 8.7, 3.2), Interp2(image8, 8.7, 3.2));
}

TEST(MCImageUtilTest, TestResample) {
  MCImageGray8 image8;
  EXPECT_TRUE(ReadMCImage<uint8_t>(FLAGS_image, &image8));
  const int rows = image8.rows(), cols = image8.cols();
  LOG(INFO) << "image8 " << rows << "x" << cols;

  const int resize_rows = rows * 0.7, resize_cols = cols * 0.7;
  MCImageGray8 resized(resize_rows, resize_cols);
  Resample(image8, &resized);
  EXPECT_TRUE(WriteMCImageGray8(resized, "test_gray8_resize.png"));
}

TEST(MCImageUtilTest, TestConvolve) {
  MCImageGray8 image8;
  EXPECT_TRUE(ReadMCImage<uint8_t>(FLAGS_image, &image8));
  const int rows = image8.rows(), cols = image8.cols();
  LOG(INFO) << "image8 " << rows << "x" << cols;

  MCImageD kernel = MCImageD::Constant(5, 1, 1.f);
//  kernel /= kernel.sum();
  BuildGaussian(1, &kernel);
LOG(INFO) << kernel;
  MCImageGray8 out(rows, cols), tmp(rows, cols);
//  Convolve<uint8_t, double, uint8_t>(image8, kernel, &out);
  LOG(INFO) << "convolute " << image8.rows() << "x" << image8.cols() << ", "
            << kernel.rows() << "x" << kernel.cols();
//  Convolve(image8, kernel, &out);
  Convolve(image8, kernel, &tmp);
  Convolve(tmp, kernel.transpose(), &out);
  LOG(INFO) << "done. " << kernel.rows() << "x" << kernel.cols();
  EXPECT_TRUE(WriteMCImageGray8(out, "test_gray8_conv.png"));

  image8.resize(15, 10);
  image8.fill(255);
  out.resize(image8.rows(), image8.cols());
  Convolve(image8, kernel, &out);
  EXPECT_TRUE(WriteMCImageGray8(out, "test_gray8_conv2.png"));
}
*/
}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}

