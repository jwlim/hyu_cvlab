// estimate_rot_test.cc
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
#include "estimate_rot.h"
#include "rvslam_util.h"
#include "rvslam_common.h"

using namespace rvslam;

class RotationMatTest {
 public:
  RotationMatTest(int, int, double, double);
  void RunTest(int test_number);
 
 private:
  void PrepareTestCase(int test_case_idx);
  double RotationError(Mat3, Mat3);

  int img_size_;
  int cube_size_;
  double min_f_, max_f_;
  double sigma_;
  int ninliers_, noutliers_; 
  double best_cost_;
  int total_;
  Mat3X pts0_;
  Mat3X pts1_;
  Mat3X points3D_;
  Mat3 IM_, R_, Ro_;
  Vec3 t_;
  std::vector<int> best_inliers_; 
  double avg_rot_err_;
  double avg_time_;
  std::ofstream record_file_;
};

RotationMatTest::RotationMatTest(
    int img_size, int cube_size, double min_f, double max_f) :  
    img_size_(img_size), cube_size_(cube_size), min_f_(min_f), max_f_(max_f) {
  IM_.setIdentity(3, 3);
  record_file_.open("eigen_rm.txt");
}

void RotationMatTest::RunTest(int test_number) {
  Vec3 total(100, 200, 500), routliers(0, 0.1, 0.5), sigma(0, 0.01, 0.05);
  double ratio, threshold;
  Mat34 Rt;
  clock_t t;

  for (int i = 0; i < total.size(); ++i) {
    total_ = total[i];
    for (int j = 0; j < routliers.size(); ++j) {
      ratio = routliers[j];
      noutliers_ = int(total_*ratio);
      ninliers_ = total_ - noutliers_;
      for (int k = 0; k < sigma.size(); ++k) {
        sigma_ = sigma[k];
        threshold = std::max(sigma_ / IM_(0, 0) * 3, 1e-4);

        record_file_ << "===========================================";
        record_file_ << "\nTotal point numbers: " << total_;
        record_file_ << "\n  Ratio of outliers: " << ratio * 100 << " %";
        record_file_ << "\n     Sigma of noise: " << sigma_;
        
        // For accurary
        avg_rot_err_ = 0;
        for (int ti = 0; ti < test_number; ++ti) {
          PrepareTestCase(ti);
          RobustEstimateRotationMat(pts0_, pts1_, &Rt, &best_inliers_, 
                                    &best_cost_, threshold);
          Ro_ = Rt.block(0, 0, 3, 3);
          avg_rot_err_ += RotationError(R_, Ro_);
        }
        avg_rot_err_ /= (double)test_number;
        record_file_ << "\n     Avg. rot. err.: " << avg_rot_err_ << " degrees";

        // For computation time
        PrepareTestCase(0);
        t = clock();
        for (int ti = 0; ti < test_number; ++ti) {
          RobustEstimateRotationMat(pts0_, pts1_, &Rt, &best_inliers_, 
                                    &best_cost_, threshold);
        }
        t = clock() - t;
        avg_time_ = 1000*((double)t)/CLOCKS_PER_SEC/test_number;
        record_file_ << "\n          Avg. time: " << avg_time_ << " ms\n\n";
      }
    }
  }
}

void RotationMatTest::PrepareTestCase(int test_case_idx) {
  points3D_.resize(3, ninliers_);
  pts0_ = Mat::Ones(3, total_);
  pts1_ = Mat::Ones(3, total_);

  // Initialize 3D points
  for (int i = 0; i < ninliers_; ++i) {
    points3D_(0, i) = (double)rand() / (RAND_MAX) * cube_size_;
    points3D_(1, i) = (double)rand() / (RAND_MAX) * cube_size_;
    points3D_(2, i) = (double)rand() / (RAND_MAX) * cube_size_ + cube_size_;
  }

  // Initialize R & t
  Vec3 r;
  r(0) = (double)rand() / (RAND_MAX) - 0.5;
  r(1) = (double)rand() / (RAND_MAX) - 0.5;
  r(2) = (double)rand() / (RAND_MAX) - 0.5;
  double theta = (double)rand() / (RAND_MAX) * (M_PI / 3);
  r = r / r.norm() * theta;
  R_ = ExpMap(r);
  t_(0) = t_(1) = t_(2) = 0; 
  
  // Initialize intrinsic matrix
  double focal = (double)rand() / (RAND_MAX) * (max_f_ - min_f_) + min_f_;
  IM_(0, 0) = IM_(1, 1) = focal;
  IM_(0, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;
  IM_(1, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;

  // Initialize inlier 2D points
  Mat34 P;
  P.setIdentity(3, 4);
  P = IM_ * P;
  pts0_.block(0, 0, 2, ninliers_)
    =   Euc(Project(P, points3D_)) 
      + (Eigen::MatrixXd::Random(2, ninliers_) * sigma_);
  P << R_, t_;
  P = IM_ * P;
  pts1_.block(0, 0, 2, ninliers_)
    =   Euc(Project(P, points3D_)) 
      + (Eigen::MatrixXd::Random(2, ninliers_) * sigma_);
  
  // Initialize outlier points
  pts0_.block(0, ninliers_, 2, noutliers_)
    = (Eigen::MatrixXd::Random(2, noutliers_).array() + 1) * img_size_ * 0.5;
  pts1_.block(0, ninliers_, 2, noutliers_)
    = (Eigen::MatrixXd::Random(2, noutliers_).array() + 1) * img_size_ * 0.5;
  
  // Prepare homogeneous coordinates
  double ifocal = 1/focal;
  pts0_.row(0) = (pts0_.row(0).array()-IM_(0, 2))*ifocal;
  pts0_.row(1) = (pts0_.row(1).array()-IM_(1, 2))*ifocal;
  pts1_.row(0) = (pts1_.row(0).array()-IM_(0, 2))*ifocal;
  pts1_.row(1) = (pts1_.row(1).array()-IM_(1, 2))*ifocal;
}

double RotationMatTest::RotationError(Mat3 R1, Mat3 R2) {
  double value = ((R1.inverse() * R2).trace() - 1) / 2.;
  value = (value < -1.0)? -1.0: (value > 1.0)? 1.0: value;
  return acos(value) * 180.0 / M_PI;
}

TEST(RelativePose, RotationMat) {
  RotationMatTest test(10, 10, 1, 3);
  test.RunTest(1000);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  return RUN_ALL_TESTS();
}
