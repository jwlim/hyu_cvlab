// Copyright (c) 2007, 2008 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

//#include <iostream>
//#include <algorithm>

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "p3p.h"
#include "rvslam_util.h"

using namespace rvslam;
using namespace std;

namespace {
/*
void MakeCamerasAndPoints(int num_cameras, int num_points,
                          Mat6X* cameras, Mat3X* points) {
  *points = Matrix3X::Random(3, num_points);
  cameras->resize(6, num_cameras);
  for (int i = 0; i < num_cameras; ++i) {
    Vec6 pose = Vec6::Random() * 2 - 1;
    pose.segment(0, 3) *= 3.141592;
    pose.segment(3, 3) *= 5.0;
    Mat34 P = MakeTransform(pose);
  }
}
*/
}  // namespace

TEST(P3P, P3PSimpleTest) {
  const int n = 3;
  Mat world_points(4, n), image_points(3, n);
  world_points << 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1;
  Mat34 P;
  P.setIdentity();
  P(2, 3) = 2.0;
  for (int i = 0; i < n; ++i) {
    image_points.col(i) = P * world_points.col(i);
    image_points.col(i) /= image_points(2, i);
  }
  LOG(ERROR) << setfill(' ') << "world_points:\n" << world_points;
  LOG(ERROR) << setfill(' ') << "P:\n" << P;
  LOG(ERROR) << setfill(' ') << "image_points:\n" << image_points;

  vector<Mat34> solutions;
  int ret = ComputePosesP3P(image_points, world_points.block(0, 0, 3, n),
                            &solutions);
  double min_err = HUGE_VAL;
  for (int i = 0; i < solutions.size(); ++i) {
    double err = 0.0;
    for (int j = 0; j < n; ++j) {
      Vec3 p = solutions[i] * world_points.col(j);
      p /= p(2);
      err += (p - image_points.col(j)).norm();
    }
    LOG(ERROR) << "solution " << i << ": err=" << err << "\n" << solutions[i];
    if (err < min_err) min_err = err;
  }
  EXPECT_NEAR(0, min_err, 1e-9);

  const double inlier_threshold = 0.01;
  vector<int> best_inliers;
  double best_score = 0.0;
  Mat34 best_model;
  EXPECT_TRUE(RobustEstimatePoseP3P(
          image_points, world_points, inlier_threshold,
          &best_model, &best_inliers, &best_score));
  EXPECT_GE(best_inliers.size(), 3);
}

/*
TEST(P3P, P3PTest2) {
  TwoViewDataSet d = TwoRealisticCameras();
  Mat34 P;
  P.block(0, 0, 3, 3) = d.R1;
  P.block(0, 3, 3, 1) = d.t1;
  LOG(ERROR) << "P:\n" << P;
  Mat3X X = d.X.block(0, 0, 3, 3);
  Mat3X x = EuclideanToHomogeneous(Project(P, X));
  vector<Mat34> solutions;
  int ret = ComputePosesP3P(x, X, &solutions);
  double min_err = HUGE_VAL;
  for (int i = 0; i < solutions.size(); ++i) {
    double err = (P - solutions[i]).norm();
    if (err < min_err) min_err = err;
    LOG(ERROR) << "solution " << i << ": err=" << err << "\n" << solutions[i];
  }
  EXPECT_NEAR(0, min_err, 1e-9);
}

TEST(P3P, RobustP3PTest) {
  TwoViewDataSet d = TwoRealisticCameras();
  Mat34 P;
  P.block(0, 0, 3, 3) = d.R1;
  P.block(0, 3, 3, 1) = d.t1;
  LOG(ERROR) << "P:\n" << P;
  Mat3X X = d.X;
  Mat3X x = EuclideanToHomogeneous(Project(P, X));

  P3PKernel kernel(x, X);
  vector<int> inliers;
  Mat34 P_est = Estimate(kernel, MLEScorer<P3PKernel>(1e-3), &inliers);
  LOG(ERROR) << "P_est: inliners=" << inliers.size() << "\n" << P_est;
  EXPECT_NEAR(FrobeniusDistance(P, P_est), 0.0, 1e-9);

  // Now make 30% of the points in x totally wrong.
  x.block(0, 0, 3, static_cast<int>(x.cols() * 0.3)).setRandom();
  P3PKernel kernel_noisy(x, X);
  P_est = Estimate(kernel_noisy, MLEScorer<P3PKernel>(1e-3), &inliers);
  LOG(ERROR) << "P__noisy: inliners=" << inliers.size() << "\n" << P_est;
  EXPECT_NEAR(FrobeniusDistance(P, P_est), 0.0, 1e-9);
}
*/
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}

