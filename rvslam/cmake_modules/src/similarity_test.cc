// similarity_test.cc
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
#include "similarity.h"
#include "rvslam_util.h"
#include "rvslam_common.h"

using namespace rvslam;

class SimilarityTest {
 public:
  SimilarityTest(int, int, double, double);
  void RunTest(int test_number);
 
 private:
  void PrepareTestCase(int test_case_idx);
  double RotationError(Mat3, Mat3);
  double TranslationError(Vec3, Vec3);
  double ScaleError(double, double);

  int img_size_;
  int cube_size_;
  double min_f_, max_f_;
  double sigma_;
  int ninliers_, noutliers_; 
  double best_cost_;
  int total_;
  Mat3X pts2d0_, pts2d1_;
  Mat3X pts3d0_, pts3d1_;
  double r_coeff_;
  double t_coeff_; 
  double s_coeff_;
  Mat3 R_, Ro_, IM_;
  Vec3 t_, to_;
  double s_, so_;
  std::vector<int> best_inliers_; 
  double avg_rot_err_;
  double avg_trans_err_;
  double avg_scale_err_;
  double avg_time_;
  std::ofstream record_file_;
};

SimilarityTest::SimilarityTest(
    int img_size, int cube_size, double min_f, double max_f) :  
    img_size_(img_size), cube_size_(cube_size), min_f_(min_f), max_f_(max_f) {
  IM_.setIdentity(3, 3);
  record_file_.open("similarity.txt");
}

void SimilarityTest::RunTest(int test_number) {
  Vec3 total(100, 200, 500), routliers(0, 0.1, 0.5), sigma(0, 0.1, 0.5);
  double ratio, threshold;
  Mat34 sRt;
  Mat3 sR;
  clock_t t;

  for (int i = 0; i < total.size(); ++i) {
    total_ = total[i];
    for (int j = 0; j < routliers.size(); ++j) {
      ratio = routliers[j];
      noutliers_ = int(total_*ratio);
      ninliers_ = total_ - noutliers_;
      for (int k = 0; k < sigma.size(); ++k) {
        sigma_ = sigma[k];
        threshold = std::max(sigma_ * 5, 1e-4);
        record_file_ << "===========================================";
        record_file_ << "\nTotal point numbers: " << total_;
        record_file_ << "\n  Ratio of outliers: " << ratio * 100 << " %";
        record_file_ << "\n     Sigma of noise: " << sigma_;

        // For accuracy
        avg_rot_err_ = 0;
        avg_trans_err_ = 0;
        avg_scale_err_ = 0;
        for (int ti = 0; ti < test_number; ++ti) {
          PrepareTestCase(ti);
          RobustEstimateSimilarity(pts2d0_, pts2d1_, pts3d0_, pts3d1_, &sRt,
                                   &best_inliers_, &best_cost_, threshold);
          sR = sRt.block(0, 0, 3, 3);
          so_ = pow(sR.determinant(), 1./3.);
          Ro_ = sR / so_;
          to_ = sRt.block(0, 3, 3, 1);
          avg_rot_err_ += RotationError(R_, Ro_);
          avg_trans_err_ += TranslationError(t_, to_);
          avg_scale_err_ += ScaleError(s_, so_);
        }
        avg_rot_err_ /= (double)test_number;
        avg_trans_err_ /= (double)test_number;
        avg_scale_err_ /= (double)test_number;
        record_file_ << "\n     Avg. rot. err.: " << avg_rot_err_ << " degrees";
        record_file_ << "\n   Avg. trans. err.: " << avg_trans_err_ << " %";
        record_file_ << "\n    Avg. scale err.: " << avg_scale_err_ << " %";

        // For computation time
        PrepareTestCase(0);
        t = clock();
        for (int ti = 0; ti < test_number; ++ti) {
          RobustEstimateSimilarity(pts2d0_, pts2d1_, pts3d0_, pts3d1_, &sRt,
                                   &best_inliers_, &best_cost_, threshold);
        }
        t = clock() - t;
        avg_time_ = 1000*((double)t)/CLOCKS_PER_SEC/test_number;
        record_file_ << "\n          Avg. time: " << avg_time_ << " ms\n\n";
      }
    }
  }
}

void SimilarityTest::PrepareTestCase(int test_case_idx) {
  pts3d0_ = Mat::Zero(3, total_);
  pts3d1_ = Mat::Zero(3, total_);
  pts2d0_ = Mat::Ones(3, total_);
  pts2d1_ = Mat::Ones(3, total_);
  
  // Initialize intrinsic matrix
  double focal = (double)rand() / (RAND_MAX) * (max_f_ - min_f_) + min_f_;
  IM_(0, 0) = IM_(1, 1) = focal;
  IM_(0, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;
  IM_(1, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;
  
  // Initialize first set of 3D points
  for (int i = 0; i < total_; ++i) {
    pts3d0_(0, i) = (double)rand() / (RAND_MAX) * cube_size_;
    pts3d0_(1, i) = (double)rand() / (RAND_MAX) * cube_size_;
    pts3d0_(2, i) = (double)rand() / (RAND_MAX) * cube_size_;
  }
  
  // Initialize R & t & s
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
  s_ = 0.5 + 0.5*(double)rand() / (RAND_MAX);

  // Initialize second set of 3D points (including outliers)
  pts3d1_ =   ((s_ * R_ * pts3d0_).colwise() + t_)
            + Mat::Random(3, total_) * sigma_;
  for (int i = ninliers_; i < total_; ++i) {
    pts3d1_(0, i) = (double)rand() / (RAND_MAX) * cube_size_;
    pts3d1_(1, i) = (double)rand() / (RAND_MAX) * cube_size_;
    pts3d1_(2, i) = (double)rand() / (RAND_MAX) * cube_size_;
  }

  // Initialize 2D points
  Mat34 P;
  P.setIdentity(3, 4);
  P = IM_ * P;
  pts2d0_.block(0, 0, 2, total_) = Euc(Project(P, pts3d0_));
  pts2d1_.block(0, 0, 2, total_) = Euc(Project(P, pts3d1_)); 

  // Prepare homogeneous coordinates
  double ifocal = 1/focal;
  pts2d0_.row(0) = (pts2d0_.row(0).array()-IM_(0, 2))*ifocal;
  pts2d0_.row(1) = (pts2d0_.row(1).array()-IM_(1, 2))*ifocal;
  pts2d1_.row(0) = (pts2d1_.row(0).array()-IM_(0, 2))*ifocal;
  pts2d1_.row(1) = (pts2d1_.row(1).array()-IM_(1, 2))*ifocal;
}

double SimilarityTest::RotationError(Mat3 R1, Mat3 R2) {
  double value = ((R1.inverse() * R2).trace() - 1) / 2.;
  value = (value < -1.0)? -1.0: (value > 1.0)? 1.0: value;
  return acos(value) * 180.0 / M_PI;
}

double SimilarityTest::TranslationError(Vec3 t1, Vec3 t2) {
  return (t1 - t2).norm() / t1.norm() * 100;
}

double SimilarityTest::ScaleError(double s1, double s2) {
  return fabs(s2 - s1) / s1 * 100;
}

TEST(RelativePose, Similarity) {
  SimilarityTest test(10, 10, 1, 3);
  test.RunTest(1000);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  return RUN_ALL_TESTS();
}
