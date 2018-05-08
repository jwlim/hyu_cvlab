// five_point_test_opencv.cc
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <vector>
#include <ctime> 
#include <cmath>
#include <iostream>
#include <fstream>
#include "rvslam_util.h"
#include "rvslam_common.h"

using namespace rvslam;

class EssentialMatAndPoseTest {
 public:
  EssentialMatAndPoseTest(int, int, double, double);
  void RunTest(int test_number);
 
 private:
  void PrepareTestCase(int test_case_idx);
  void Evaluation(int test_case_idx);
  double RotationError(Mat3, Mat3);
  double TranslationError(Vec3, Vec3);

  int img_size_;
  int cube_size_;
  double min_f_, max_f_;
  double sigma_;
  int ninliers_;
  int noutliers_;
  int total_;
  Mat2X pts0_;
  Mat2X pts1_;
  Mat3X points3D_;
  Mat3 IM_, R_, Ro_, E_, Eo_;
  Vec3 t_, to_;
  Eigen::Array<bool, 1, Eigen::Dynamic> mask_;
  double avg_rot_err_;
  double avg_trans_err_;
  double avg_time_;
  std::ofstream record_file_;
 
  // for opencv
  void PrepareCVData();
  void ObtainCVData();
  cv::Mat cv_input0_, cv_input1_;
  double cv_focal_;
  cv::Point2d cv_pp_;
  int cv_method_;
  cv::Mat cv_E_, cv_mask_, cv_R_, cv_t_;
};

EssentialMatAndPoseTest::EssentialMatAndPoseTest(
    int img_size, int cube_size, double min_f, double max_f) :  
    img_size_(img_size), cube_size_(cube_size), min_f_(min_f), max_f_(max_f) {
  IM_.setIdentity(3, 3);
  record_file_.open("opencv_fp.txt");
}

void EssentialMatAndPoseTest::RunTest(int test_number) {
  Vec3 total(100, 200, 500), routliers(0, 0.1, 0.5), sigma(0, 0.01, 0.05);
  double ratio, threshold;

  for (int i = 0; i < total.size(); ++i) {
    total_ = total[i];
    for (int j = 0; j < routliers.size(); ++j) {
      ratio = routliers[j];
      noutliers_ = int(total_*ratio);
      ninliers_ = total_ - noutliers_;
      for (int k = 0; k < sigma.size(); ++k) {
        sigma_ = sigma[k];
        threshold = MAX(sigma_ * 1.3, 1e-4), cv_mask_;
        record_file_ << "===========================================";
        record_file_ << "\nTotal point numbers: " << total_;
        record_file_ << "\n  Ratio of outliers: " << ratio * 100 << " %";
        record_file_ << "\n     Sigma of noise: " << sigma_;
        
        // for accurary
        avg_rot_err_ = 0;
        avg_trans_err_ = 0;
        for (int ti = 0; ti < test_number; ++ti) {
          PrepareTestCase(ti);
          PrepareCVData();
          cv_E_ = cv::findEssentialMat(cv_input0_, cv_input1_, cv_focal_,
                                     cv_pp_, cv_method_, 0.99, threshold,
                                     cv_mask_);
          if (cv_E_.rows > 3) {
            int count = cv_E_.rows / 3;
            int row = (rand() % count) * 3;
            cv_E_ = cv_E_.rowRange(row, row + 3) * 1.0;
          }
          cv_E_ /= cv_E_.at<double>(2, 2);

          cv_mask_ = cv_mask_.t();
          cv::recoverPose(cv_E_, cv_input0_, cv_input1_, cv_R_, cv_t_,
                          cv_focal_, cv_pp_, cv_mask_);
          ObtainCVData();
          Evaluation(ti);
        }
        avg_rot_err_ /= test_number;
        avg_trans_err_ /= test_number;
        record_file_ << "\n     Avg. rot. err.: " << avg_rot_err_ << " degrees";
        record_file_ << "\n   Avg. trans. err.: " << avg_trans_err_ << " %";
        
        // for computation time
        clock_t t;
        PrepareTestCase(0);
        PrepareCVData();
        t = clock();
        for (int ti = 0; ti < test_number; ++ti) {
          cv_E_ = cv::findEssentialMat(cv_input0_, cv_input1_, cv_focal_,
                                      cv_pp_, cv_method_, 0.99, threshold,
                                      cv_mask_);
          if (cv_E_.rows > 3) {
            int count = cv_E_.rows / 3;
            int row = (rand() % count) * 3;
            cv_E_ = cv_E_.rowRange(row, row + 3) * 1.0;
          }
          cv_mask_ = cv_mask_.t();
          cv::recoverPose(cv_E_, cv_input0_, cv_input1_, cv_R_, cv_t_,
                          cv_focal_, cv_pp_, cv_mask_);
          cv_mask_ = cv_mask_.t();
        }
        t = clock() - t;
        avg_time_ = 1000*((double)t)/CLOCKS_PER_SEC/test_number;
        record_file_ << "\n          Avg. time: " << avg_time_ << " ms\n\n";
      }
    }
  }
}

void EssentialMatAndPoseTest::PrepareTestCase(int test_case_idx) {
  points3D_.resize(3, ninliers_);
  pts0_.resize(2, total_);
  pts1_.resize(2, total_);
  mask_.resize(1, total_);

  // initialize 3D points
  for (int i = 0; i < ninliers_; ++i) {
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
  E_ = CrossProductMatrix(t_) * R_;
  
  // initialize intrinsic matrix
  double focal = (double)rand() / (RAND_MAX) * (max_f_ - min_f_) + min_f_;
  IM_(0, 0) = IM_(1, 1) = focal;
  IM_(0, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;
  IM_(1, 2) = (img_size_ * 0.5 + (double)rand() / (RAND_MAX) * 4. - 2.) * focal;

  // initialize inlier 2D points
  Mat2X noise;
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
  
  // initialize outlier points
  pts0_.block(0, ninliers_, 2, noutliers_)
    = (Eigen::MatrixXd::Random(2, noutliers_).array() + 1) * img_size_ * 0.5;
  pts1_.block(0, ninliers_, 2, noutliers_)
    = (Eigen::MatrixXd::Random(2, noutliers_).array() + 1) * img_size_ * 0.5;
}

void EssentialMatAndPoseTest::Evaluation(int test_case_idx) {
  avg_rot_err_ += RotationError(R_, Ro_);
  avg_trans_err_ += TranslationError(t_, to_);
}

double EssentialMatAndPoseTest::RotationError(Mat3 R1, Mat3 R2) {
  double value = ((R1.inverse() * R2).trace() - 1) / 2.;
  value = (value < -1.0)? -1.0: (value > 1.0)? 1.0: value;
  return acos(value) * 180.0 / M_PI;
}

double EssentialMatAndPoseTest::TranslationError(Vec3 t1, Vec3 t2) {
  t1 /= t1.norm();
  t2 /= t2.norm();
  return (t1-t2).norm() * 100;
}

void EssentialMatAndPoseTest::PrepareCVData() {
  cv_input0_.create(total_, 2, CV_64F);
  cv_input1_.create(total_, 2, CV_64F);
  for (int i = 0; i < total_; i++) {
    cv_input0_.at<double>(i, 0) = pts0_(0, i);
    cv_input0_.at<double>(i, 1) = pts0_(1, i);
    cv_input1_.at<double>(i, 0) = pts1_(0, i);
    cv_input1_.at<double>(i, 1) = pts1_(1, i);
  }
  cv_focal_ = IM_(0, 0);
  cv_pp_ = cv::Point2d(IM_(0, 2), IM_(1, 2));
  cv_method_ = cv::RANSAC;
  cv_mask_.create(1, total_, CV_8U);
  cv_E_.create(3, 3, CV_64F);
  cv_R_.create(3, 3, CV_64F);
  cv_t_.create(3, 1, CV_64F);
}

void EssentialMatAndPoseTest::ObtainCVData() {
  Eo_ << cv_E_.at<double>(0, 0), cv_E_.at<double>(0, 1), cv_E_.at<double>(0, 2),
         cv_E_.at<double>(1, 0), cv_E_.at<double>(1, 1), cv_E_.at<double>(1, 2),
         cv_E_.at<double>(2, 0), cv_E_.at<double>(2, 1), cv_E_.at<double>(2, 2);
          
  Ro_ << cv_R_.at<double>(0, 0), cv_R_.at<double>(0, 1), cv_R_.at<double>(0, 2),
         cv_R_.at<double>(1, 0), cv_R_.at<double>(1, 1), cv_R_.at<double>(1, 2),
         cv_R_.at<double>(2, 0), cv_R_.at<double>(2, 1), cv_R_.at<double>(2, 2);
          
  to_ << cv_t_.at<double>(0, 0), cv_t_.at<double>(1, 0), cv_t_.at<double>(2, 0);
  
  for (int i = 0; i < total_; ++i)
    mask_(i) = cv_mask_.at<unsigned char>(i, 0) != 0;
}

TEST(RelativePose, EssentialMatAndPose) {
  EssentialMatAndPoseTest test(10, 10, 1, 3);
  test.RunTest(100);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  return RUN_ALL_TESTS();
}
