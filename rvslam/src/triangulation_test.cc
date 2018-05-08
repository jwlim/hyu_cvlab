// triangulation_test.cc
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime> 
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "triangulation.h"
#include "rvslam_util.h"
#include "rvslam_common.h"

using namespace rvslam;

class TriangulationTest {
 public:
  TriangulationTest(int, int, double, double);
  void RunTest(int test_number);
 
 private:
  void PrepareTestCase(int test_case_idx);
  double PointsError(Mat3X, Eigen::ArrayXXd);

  int img_size_;
  int cube_size_;
  double min_f_, max_f_;
  double sigma_;
  int total_;
  Mat3X pts0_, pts1_;
  Mat34 pose0_, pose1_;
  Mat3X points3D_;
  Eigen::ArrayXXd points4D_;
  Mat3 IM_, R_;
  Vec3 t_;
  double avg_pts_err_;
  double avg_time_;
  std::ofstream record_file_;
};

TriangulationTest::TriangulationTest(
    int img_size, int cube_size, double min_f, double max_f) :  
    img_size_(img_size), cube_size_(cube_size), min_f_(min_f), max_f_(max_f) {
  IM_.setIdentity(3, 3);
  record_file_.open("eigen_trian.txt");
}

void TriangulationTest::RunTest(int test_number) {
  Vec3 total(100, 200, 500), sigma(0, 0.01, 0.05);
  clock_t t;

  for (int i = 0; i < total.size(); ++i) {
    total_ = total[i];
    for (int j = 0; j < sigma.size(); ++j) {
      sigma_ = sigma[j];
      record_file_ << "===========================================";
      record_file_ << "\nTotal point numbers: " << total_;
      record_file_ << "\n     Sigma of noise: " << sigma_;
      
      // for accurary
      avg_pts_err_ = 0;
      for (int ti = 0; ti < test_number; ++ti) {
        PrepareTestCase(ti);
        TriangulatePoints(pose0_, pose1_, pts0_, pts1_, &points4D_);
        avg_pts_err_ += PointsError(points3D_, points4D_);
      }
      avg_pts_err_ /= (double)test_number;
      record_file_ << "\n   Avg. points err.: " << avg_pts_err_ << " units";

      // for computation time
      PrepareTestCase(0);
      t = clock();
      for (int ti = 0; ti < test_number; ++ti) {
        TriangulatePoints(pose0_, pose1_, pts0_, pts1_, &points4D_);
      }
      t = clock() - t;
      avg_time_ = 1000*((double)t)/CLOCKS_PER_SEC/test_number;
      record_file_ << "\n          Avg. time: " << avg_time_ << " ms\n\n";
    }
  }
}

void TriangulationTest::PrepareTestCase(int test_case_idx) {
  points3D_.resize(3, total_);
  points4D_.resize(4, total_);
  pts0_ = Mat::Constant(3, total_, 1);
  pts1_ = Mat::Constant(3, total_, 1);

  // initialize 3D points
  for (int i = 0; i < total_; ++i) {
    points3D_(0, i) = (double)rand() / (RAND_MAX) * cube_size_;
    points3D_(1, i) = (double)rand() / (RAND_MAX) * cube_size_;
    points3D_(2, i) = (double)rand() / (RAND_MAX) * cube_size_;
  }

  // initialize R & t
  Vec3 r;
  r(0) = (double)rand() / (RAND_MAX) - 0.5;
  r(1) = (double)rand() / (RAND_MAX) - 0.5;
  r(2) = (double)rand() / (RAND_MAX) - 0.5;
  double theta = (double)rand() / (RAND_MAX) * (M_PI / 2);
  r = r / r.norm() * theta;
  R_ = ExpMap(r);
  t_(0) = (double)rand() / (RAND_MAX) * cube_size_;
  t_(1) = (double)rand() / (RAND_MAX) * cube_size_;
  t_(2) = (double)rand() / (RAND_MAX) * cube_size_ + cube_size_;
  
  // initialize intrinsic matrix
  double focal = (double)rand() / (RAND_MAX) * (max_f_ - min_f_) + min_f_;
  IM_(0, 0) = IM_(1, 1) = focal;
  IM_(0, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;
  IM_(1, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;

  // initialize inlier 2D points
  Mat2X noise;
  Mat34 P;
  P.setIdentity(3, 4);
  pose0_ = P;
  P = IM_ * P;
  pts0_.block(0, 0, 2, total_)
    =   Euc(Project(P, points3D_)) 
      + (Eigen::MatrixXd::Random(2, total_) * sigma_);
  P << R_, t_;
  pose1_ = P;
  P = IM_ * P;
  pts1_.block(0, 0, 2, total_)
    =   Euc(Project(P, points3D_)) 
      + (Eigen::MatrixXd::Random(2, total_) * sigma_);
  
  // prepare homogeneous coordinates
  double ifocal = 1/focal;
  pts0_.row(0) = (pts0_.row(0).array()-IM_(0, 2))*ifocal;
  pts0_.row(1) = (pts0_.row(1).array()-IM_(1, 2))*ifocal;
  pts1_.row(0) = (pts1_.row(0).array()-IM_(0, 2))*ifocal;
  pts1_.row(1) = (pts1_.row(1).array()-IM_(1, 2))*ifocal;
}

double TriangulationTest::PointsError(Mat3X pts3d, Eigen::ArrayXXd pts4d) {
  Mat3X diff = pts3d - pts4d.matrix().block(0, 0, 3, total_);
  double sum = diff.colwise().norm().sum();
  return sum / total_;
}

TEST(Geometry, Triangulation) {
  TriangulationTest test(10, 10, 1, 3);
  test.RunTest(100);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  return RUN_ALL_TESTS();
}
