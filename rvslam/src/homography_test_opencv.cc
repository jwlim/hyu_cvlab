// homography_test_opencv.cc
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

class HomographyAndPoseTest {
 public:
  HomographyAndPoseTest(int, int, double, double);
  void RunTest(int test_number);
 
 private:
  void PrepareTestCase(int test_case_idx);
  double HomographyError(Mat3, Mat3);
  double RotationError(Mat3, Mat3);
  double TranslationError(Vec3, Vec3);

  int img_size_;
  int cube_size_;
  double min_f_, max_f_;
  double sig_;
  int ninliers_;
  int noutliers_;
  int total_;
  Mat3X pts0_;
  Mat3X pts1_;
  Mat3X points3D_;
  Mat3 IM_, R_, Ro_, H_, Ho_;
  Vec3 t_, to_;
  Eigen::Array<bool, 1, Eigen::Dynamic> mask_;
  double avg_homo_err_;
  double avg_rot_err_;
  double avg_trans_err_;
  double avg_time_;
  std::ofstream record_file_;
 
  // for opencv
  void PrepareCVData();
  void ObtainCVData();
  std::vector<cv::Point2f> cvpts0_, cvpts1_;
  double cv_focal_;
  cv::Point2d cv_pp_;
  int cv_method_;
  cv::Mat cv_H_, cv_mask_;
  std::vector<cv::Mat> Rs_, ts_;
};

HomographyAndPoseTest::HomographyAndPoseTest(
    int img_size, int cube_size, double min_f, double max_f) :  
    img_size_(img_size), cube_size_(cube_size), min_f_(min_f), max_f_(max_f) {
  IM_.setIdentity(3, 3);
  record_file_.open("opencv_hm.txt");
}

void HomographyAndPoseTest::RunTest(int test_number) {
  Vec3 total(100, 200, 500), routliers(0, 0.1, 0.5), sigma(0, 0.01, 0.05);
  double ratio, threshold;
  cv::Mat I = cv::Mat::eye(3, 3, CV_64F);

  for (int i = 0; i < total.size(); ++i) {
    total_ = total[i];
    for (int j = 0; j < routliers.size(); ++j) {
      ratio = routliers[j];
      noutliers_ = int(total_*ratio);
      ninliers_ = total_ - noutliers_;
      for (int k = 0; k < sigma.size(); ++k) {
        sig_ = sigma[k];
        threshold = MAX(sig_ / IM_(0, 0) * 4, 1e-4);
        record_file_ << "===========================================";
        record_file_ << "\nTotal point numbers: " << total_;
        record_file_ << "\n  Ratio of outliers: " << ratio * 100 << " %";
        record_file_ << "\n     Sigma of noise: " << sig_;
        
        // for accurary
        avg_homo_err_ = 0;
        avg_rot_err_ = 0;
        avg_trans_err_ = 0;
        for (int ti = 0; ti < test_number; ++ti) {
          PrepareTestCase(ti);
          PrepareCVData();
          cv_H_ = cv::findHomography(cvpts0_, cvpts1_, cv_method_,
                                     threshold, cv_mask_);
          decomposeHomographyMat(cv_H_, I, Rs_, ts_, cv::noArray());
          ObtainCVData();
          avg_homo_err_ += HomographyError(H_, Ho_);
          avg_rot_err_ += RotationError(R_, Ro_);
          avg_trans_err_ += TranslationError(t_, to_);
        }
        avg_homo_err_ /= test_number;
        avg_rot_err_ /= test_number;
        avg_trans_err_ /= test_number;
        record_file_ << "\n    Avg. homo. err.: " << avg_homo_err_ << " %";
        record_file_ << "\n     Avg. rot. err.: " << avg_rot_err_ << " degrees";
        record_file_ << "\n   Avg. trans. err.: " << avg_trans_err_ << " %";
        
        // for computation time
        clock_t t;
        PrepareTestCase(0);
        PrepareCVData();
        t = clock();
        for (int ti = 0; ti < test_number; ++ti) {
          cv_H_ = cv::findHomography(cvpts0_, cvpts1_, cv_method_,
                                     threshold, cv_mask_);
          decomposeHomographyMat(cv_H_, I, Rs_, ts_, cv::noArray());
        }
        t = clock() - t;
        avg_time_ = 1000*((double)t)/CLOCKS_PER_SEC/test_number;
        record_file_ << "\n          Avg. time: " << avg_time_ << " ms\n\n";
      }
    }
  }
}

void HomographyAndPoseTest::PrepareTestCase(int test_case_idx) {
  points3D_.resize(3, ninliers_);
  pts0_ = Mat::Constant(3, total_, 1);
  pts1_ = Mat::Constant(3, total_, 1);
  mask_.resize(1, total_);

  // initialize 3D points
  double z_value = (double)rand() / (RAND_MAX) * cube_size_;
  for (int i = 0; i < ninliers_; ++i) {
    points3D_(0, i) = (double)rand() / (RAND_MAX) * cube_size_;
    points3D_(1, i) = (double)rand() / (RAND_MAX) * cube_size_;
    points3D_(2, i) = z_value;
  }

  // initialize R & t & H
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
  H_ << R_(0, 0), R_(0, 1), R_(0, 2) + t_(0) / z_value,
        R_(1, 0), R_(1, 1), R_(1, 2) + t_(1) / z_value,
        R_(2, 0), R_(2, 1), R_(2, 2) + t_(2) / z_value;
  
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
      + (Eigen::MatrixXd::Random(2, ninliers_) * sig_);
  P << R_, t_;
  P = IM_ * P;
  pts1_.block(0, 0, 2, ninliers_)
    =   Euc(Project(P, points3D_)) 
      + (Eigen::MatrixXd::Random(2, ninliers_) * sig_);
  
  // initialize outlier points
  pts0_.block(0, ninliers_, 2, noutliers_)
    = (Eigen::MatrixXd::Random(2, noutliers_).array() + 1) * img_size_ * 0.5;
  pts1_.block(0, ninliers_, 2, noutliers_)
    = (Eigen::MatrixXd::Random(2, noutliers_).array() + 1) * img_size_ * 0.5;
  
  // prepare homogeneous coordinates
  double ifocal = 1/focal;
  pts0_.row(0) = (pts0_.row(0).array()-IM_(0, 2))*ifocal;
  pts0_.row(1) = (pts0_.row(1).array()-IM_(1, 2))*ifocal;
  pts1_.row(0) = (pts1_.row(0).array()-IM_(0, 2))*ifocal;
  pts1_.row(1) = (pts1_.row(1).array()-IM_(1, 2))*ifocal;
}

double HomographyAndPoseTest::HomographyError(Mat3 H1, Mat3 H2) {
  H1 /= H1(2, 2);
  H1 /= H1.norm();
  H2 /= H2(2, 2);
  H2 /= H2.norm();
  return (H1-H2).norm() * 100;
}

double HomographyAndPoseTest::RotationError(Mat3 R1, Mat3 R2) {
  double value = ((R1.inverse() * R2).trace() - 1) / 2.;
  value = (value < -1.0)? -1.0: (value > 1.0)? 1.0: value;
  return acos(value) * 180.0 / M_PI;
}

double HomographyAndPoseTest::TranslationError(Vec3 t1, Vec3 t2) {
  t1 /= t1.norm();
  t2 /= t2.norm();
  return (t1-t2).norm() * 100;
}

void HomographyAndPoseTest::PrepareCVData() {
  cvpts0_.resize(total_);
  cvpts1_.resize(total_);
  for (int i = 0; i < total_; i++) {
    cvpts0_[i].x = pts0_(0, i);
    cvpts0_[i].y = pts0_(1, i);
    cvpts1_[i].x = pts1_(0, i);
    cvpts1_[i].y = pts1_(1, i);
  }
  cv_focal_ = IM_(0, 0);
  cv_pp_ = cv::Point2d(IM_(0, 2), IM_(1, 2));
  cv_method_ = cv::RANSAC;
  cv_mask_.create(1, total_, CV_8U);
  cv_H_.create(3, 3, CV_64F);
}

void HomographyAndPoseTest::ObtainCVData() {
  Ho_ << cv_H_.at<double>(0, 0), cv_H_.at<double>(0, 1), cv_H_.at<double>(0, 2),
         cv_H_.at<double>(1, 0), cv_H_.at<double>(1, 1), cv_H_.at<double>(1, 2),
         cv_H_.at<double>(2, 0), cv_H_.at<double>(2, 1), cv_H_.at<double>(2, 2);

  Mat3 R;
  double e, min = 1e10;
  for (int i = 0; i < Rs_.size(); ++i) {
    cv::Mat Ri = Rs_[i]; 
    R << Ri.at<double>(0, 0), Ri.at<double>(0, 1), Ri.at<double>(0, 2),
         Ri.at<double>(1, 0), Ri.at<double>(1, 1), Ri.at<double>(1, 2),
         Ri.at<double>(2, 0), Ri.at<double>(2, 1), Ri.at<double>(2, 2);
    if (TranslationError(R.col(0).cross(R.col(1)), R.col(2)) > 100)
      R.col(2) = -R.col(2);
    if (RotationError(R, R_) > 170) 
      R << -R.col(0), -R.col(1), R.col(2);
    e = RotationError(R_, R);
    if (e < min) {
      Ro_ = R;
      min = e;
    }
  }
  
  Vec3 t;
  min = 1e10;
  for (int i = 0; i < ts_.size(); ++i) {
    cv::Mat ti = ts_[i]; 
    t << ti.at<double>(0, 0), ti.at<double>(1, 0), ti.at<double>(2, 0);
    e = TranslationError(t_, t);
    if (e < min) {
      to_ = t;
      min = e;
    }
  }

  for (int i = 0; i < total_; ++i)
    mask_(i) = cv_mask_.at<unsigned char>(i, 0) != 0;
}

TEST(RelativePose, HomographyAndPose) {
  HomographyAndPoseTest test(10, 10, 1, 3);
  test.RunTest(100);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  return RUN_ALL_TESTS();
}
