#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "image_file.h"
//#include "image_type.h"
#include "image_util.h"

using namespace std;
using namespace rvslam;

//DEFINE_string(image, "test_gray8.png", "Test image path.");

namespace {

typedef Eigen::ArrayXXf ArrayF;

void TestGaussian1D(const ArrayF& kernel, const double sigma) {
  int rad = kernel.cols() / 2;
  for (int i = 1; i < rad; ++i) {
    const double coef = -1 / (2 * sigma * sigma);
    const double ratio = exp(coef * i * i);
    EXPECT_NEAR(ratio, kernel(0, rad - i) / kernel(0, rad), 1e-6);
    EXPECT_NEAR(ratio, kernel(0, rad + i) / kernel(0, rad), 1e-6);
  }
}

TEST(ImageUtilTest, FontMapDump) {
  Image<unsigned char> img_char;
  img_char.FromBuffer(_g_fontimgdata, 480, 8);

  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 480; ++j)
      img_char(j,i) = img_char(j,i) * 255;

  WriteImageGray8(img_char, "font.png");
}

TEST(ImageUtilTest, FontHelloWorld) {
  MCImage<uint8_t, 3> img_rgb;
  img_rgb.Resize(100,100);
  img_rgb.fill(255);

  DrawText(img_rgb, 10, 20, MCImageRGB8::MakePixel(255, 0, 0), "Hello World!\0");
  DrawTextFormat(img_rgb, 10, 50, MCImageRGB8::MakePixel(0, 255, 0), "Frame %04d", 2);
  WriteImageRGB8(img_rgb, "font-test.png");
}

TEST(ImageUtilTest, TestBuildGaussian1D) {
  ArrayF kernel(1, 9);
  double sigma = 1.0;
  BuildGaussian1D(sigma, &kernel);
  TestGaussian1D(kernel, sigma);
  sigma = 0.5;
  BuildGaussian1D(sigma, &kernel);
  TestGaussian1D(kernel, sigma);
  sigma = 1.5;
  BuildGaussian1D(sigma, &kernel);
  TestGaussian1D(kernel, sigma);
  kernel.resize(1, 13);
  BuildGaussian1D(sigma, &kernel);
  TestGaussian1D(kernel, sigma);
  sigma = 2.5;
  BuildGaussian1D(sigma, &kernel);
  TestGaussian1D(kernel, sigma);
}

TEST(ImageUtilTest, TestConvolve) {
  ArrayF kernel(5, 1);
  BuildGaussian1D(0.5, &kernel);
//  LOG(INFO) << setfill(' ') << endl << kernel;

  ArrayF imgf(10, 10), out(10, 10);
  imgf.fill(1.0);
  for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 5; ++x) {
      imgf(x, y + 5) = imgf(x + 5, y) = 0.0;
    }
  }
//  LOG(INFO) << setfill(' ') << endl << imgf;
//  LOG(INFO) << "convolute " << imgf.rows() << "x" << imgf.cols() << ", "
//      << kernel.rows() << "x" << kernel.cols();
  Convolve(imgf, kernel, &out);
//  LOG(INFO) << setfill(' ') << endl << out;
  ArrayF ref(10, 1);
  ref << 0.893285, 0.999736, 1.000000, 0.999736, 0.893285, 0.106715, 0.000264,
      0.000000, 0.000000, 0.000000;
  for (int y = 0; y < imgf.cols() / 2; ++y) {
    for (int x = 0; x < imgf.rows(); ++x) {
      EXPECT_NEAR(ref(x), out(x, y), 1e-5);
      EXPECT_NEAR(ref(imgf.rows() - 1 - x), out(x, y + imgf.cols() / 2), 1e-5);
    }
  }
  ArrayF ref2d(10, 10), out2d(10, 10);
  Convolve(out, kernel.transpose(), &out2d);
//  LOG(INFO) << setfill(' ') << endl << out2d;
  ref2d << 0.797959, 0.893050, 0.893285, 0.893050, 0.797959, 0.095327, 0.000236,
        0.000000, 0.000000, 0.000000,
        0.893050, 0.999472, 0.999736, 0.999472, 0.893050, 0.106686, 0.000264,
        0.000000, 0.000000, 0.000000,
        0.893285, 0.999736, 1.000000, 0.999736, 0.893285, 0.106715, 0.000264,
        0.000000, 0.000000, 0.000000,
        0.893050, 0.999472, 0.999736, 0.999472, 0.893078, 0.106922, 0.000528,
        0.000264, 0.000264, 0.000236,
        0.797959, 0.893050, 0.893285, 0.893078, 0.809347, 0.190653, 0.106922,
        0.106715, 0.106686, 0.095327,
        0.095327, 0.106686, 0.106715, 0.106922, 0.190653, 0.809347, 0.893078,
        0.893285, 0.893050, 0.797959,
        0.000236, 0.000264, 0.000264, 0.000528, 0.106922, 0.893078, 0.999472,
        0.999736, 0.999472, 0.893050,
        0.000000, 0.000000, 0.000000, 0.000264, 0.106715, 0.893285, 0.999736,
        1.000000, 0.999736, 0.893285,
        0.000000, 0.000000, 0.000000, 0.000264, 0.106686, 0.893050, 0.999472,
        0.999736, 0.999472, 0.893050,
        0.000000, 0.000000, 0.000000, 0.000236, 0.095327, 0.797959, 0.893050,
        0.893285, 0.893050, 0.797959;
  for (int y = 0; y < imgf.cols(); ++y) {
    for (int x = 0; x < imgf.rows(); ++x) {
      EXPECT_NEAR(ref2d(x, y), out2d(x, y), 1e-5);
    }
  }
  ArrayF derivative(3, 1), imgx(10, 10), imgy(10, 10);
  derivative << -1.0, 0, 1.0;
  Convolve(out2d, derivative, &imgx);
  Convolve(out2d, derivative.transpose(), &imgy);
//  LOG(INFO) << setfill(' ') << endl << out2d;
//  LOG(INFO) << setfill(' ') << endl << imgx;
//  LOG(INFO) << setfill(' ') << endl << imgy;
  for (int y = 1; y < imgf.cols() - 1; ++y) {
    for (int x = 1; x < imgf.rows() - 1; ++x) {
      EXPECT_NEAR(out2d(x - 1, y) - out2d(x + 1, y), imgx(x, y), 1e-5);
      EXPECT_NEAR(out2d(x, y - 1) - out2d(x, y + 1), imgy(x, y), 1e-5);
    }
  }
}

/*
uint8_t Interp2Ref(const Image8& image, double x, double y) {
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

TEST(ImageUtilTest, TestInterp2) {
  Image8 image8(10, 10);
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

TEST(ImageUtilTest, TestResample) {
  Image8 image8;
  EXPECT_TRUE(ReadImage<uint8_t>(FLAGS_image, &image8));
  const int rows = image8.rows(), cols = image8.cols();
  LOG(INFO) << "image8 " << rows << "x" << cols;

  const int resize_rows = rows * 0.7, resize_cols = cols * 0.7;
  Image8 resized(resize_rows, resize_cols);
  Resample(image8, &resized);
  EXPECT_TRUE(WriteImage8(resized, "test_gray8_resize.png"));
}

TEST(ImageUtilTest, TestConvolve) {
  Image8 image8;
  EXPECT_TRUE(ReadImage<uint8_t>(FLAGS_image, &image8));
  const int rows = image8.rows(), cols = image8.cols();
  LOG(INFO) << "image8 " << rows << "x" << cols;

  ImageD kernel = ImageD::Constant(5, 1, 1.f);
//  kernel /= kernel.sum();
  BuildGaussian(1, &kernel);
LOG(INFO) << kernel;
  Image8 out(rows, cols), tmp(rows, cols);
//  Convolve<uint8_t, double, uint8_t>(image8, kernel, &out);
  LOG(INFO) << "convolute " << image8.rows() << "x" << image8.cols() << ", "
            << kernel.rows() << "x" << kernel.cols();
//  Convolve(image8, kernel, &out);
  Convolve(image8, kernel, &tmp);
  Convolve(tmp, kernel.transpose(), &out);
  LOG(INFO) << "done. " << kernel.rows() << "x" << kernel.cols();
  EXPECT_TRUE(WriteImage8(out, "test_gray8_conv.png"));

  image8.resize(15, 10);
  image8.fill(255);
  out.resize(image8.rows(), image8.cols());
  Convolve(image8, kernel, &out);
  EXPECT_TRUE(WriteImage8(out, "test_gray8_conv2.png"));
}
*/
}  // namespace

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}

