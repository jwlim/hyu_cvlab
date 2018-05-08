// sim3_test.cc
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
#include "sim3.h"
#include "rvslam_util.h"
#include "rvslam_common.h"

using namespace rvslam;

class Sim3Test {
 public:
  Sim3Test();
  void RunTest(int test_number);
 
 private:
  void PrepareTestCase(int test_case_idx);
  double RotationError(Mat3, Mat3);
  double TranslationError(Vec3, Vec3);
  double ScaleError(double, double);
  double HomogeneousError(Mat4, Mat4);

  double r_coeff_;
  double t_coeff_; 
  double s_coeff_;
  Mat3 R_, Ro_;
  Vec3 t_, to_;
  double s_, so_;
  double avg_rot_err_;
  double avg_trans_err_;
  double avg_scale_err_;
  double avg_homo_err_;
  double avg_vec_err_;
  std::ofstream record_file_;
};

Sim3Test::Sim3Test() {
  record_file_.open("sim3.txt");
}

void Sim3Test::RunTest(int test_number) {
  Vec4 r_coeffs(1, 1e2, 1e-3, 1e-6);
  Vec4 t_coeffs(1, 1e2, 1e-3, 1e-6);
  Vec4 s_coeffs(1, 1e2, 1e-3, 1e-6);

  for (int i = 0; i < r_coeffs.size(); ++i) {
    r_coeff_ = r_coeffs[i];
    for (int j = 0; j < t_coeffs.size(); ++j) {
      t_coeff_ = t_coeffs[j];
      for (int k = 0; k < s_coeffs.size(); ++k) {
        s_coeff_ = s_coeffs[k];
        record_file_ << "===========================================";
        record_file_ << "\n    Rotation Coeff.: " << r_coeff_;
        record_file_ << "\n Translation Coeff.: " << t_coeff_;
        record_file_ << "\n       Scale Coeff.: " << s_coeff_;
        avg_rot_err_ = 0;
        avg_trans_err_ = 0;
        avg_scale_err_ = 0;
        avg_homo_err_ = 0;
        avg_vec_err_ = 0;
        for (int ti = 0; ti < test_number; ++ti) {
          PrepareTestCase(ti);
          // Stage 1: ExpMap, MatLog, Inverse, Operator*
          Sim3 S1(R_, t_, s_);
          Sim3 S2 = Sim3(S1.Log()).Inverse();
          S2 *= S1 * S1;
          Ro_ = S2.Rotation().toRotationMatrix();
          to_ = S2.Translation();
          so_ = S2.Scale();
          avg_rot_err_ += RotationError(R_, Ro_);
          avg_trans_err_ += TranslationError(t_, to_);
          avg_scale_err_ += ScaleError(s_, so_);
          
          // Stage 2: Homogeneous, Transformation, Adjoint
          S2 = S1.Inverse();
          Mat7 A1 = S1.Adj();
          Mat4 H1 = S1.HomogeneousMat();
          Mat4 H_ori = Sim3((Vec7)(A1 * S2.Log())).HomogeneousMat();
          Mat4 H_new = H1 * Sim3(S2.Log()).HomogeneousMat() * H1.inverse();
          avg_homo_err_ += HomogeneousError(H_ori, H_new);

          // Stage 3: Operator*
          Vec3 v1 = S1 * t_;
          Vec3 v2 = (H1 * Vec4(t_(0), t_(1), t_(2), 1)).head(3);
          avg_vec_err_ += TranslationError(v1, v2);
        }
        avg_rot_err_ /= (double)test_number;
        avg_trans_err_ /= (double)test_number;
        avg_scale_err_ /= (double)test_number;
        avg_homo_err_ /= (double)test_number;
        avg_vec_err_ /= (double)test_number;
        record_file_ << "\n     Avg. rot. err.: " << avg_rot_err_ << " degrees";
        record_file_ << "\n   Avg. trans. err.: " << avg_trans_err_ << " %";
        record_file_ << "\n    Avg. scale err.: " << avg_scale_err_ << " %";
        record_file_ << "\n    Avg. homo. err.: " << avg_homo_err_ << " %";
        record_file_ << "\n     Avg. vec. err.: " << avg_vec_err_ << " %\n\n";
      }
    }
  }
}

void Sim3Test::PrepareTestCase(int test_case_idx) {
  // initialize R & t & s
  Vec3 r;
  r(0) = (double)rand() / (RAND_MAX) - 0.5;
  r(1) = (double)rand() / (RAND_MAX) - 0.5;
  r(2) = (double)rand() / (RAND_MAX) - 0.5;
  r *= r_coeff_;
  R_ = ExpMap(r);
  t_(0) = (double)rand() / (RAND_MAX) - 0.5;
  t_(1) = (double)rand() / (RAND_MAX) - 0.5;
  t_(2) = (double)rand() / (RAND_MAX) - 0.5;
  t_ *= t_coeff_;
  s_ = (double)rand() / (RAND_MAX) + 1e-9;
  s_ *= s_coeff_;
}

double Sim3Test::RotationError(Mat3 R1, Mat3 R2) {
  double value = ((R1.inverse() * R2).trace() - 1) / 2.;
  value = (value < -1.0)? -1.0: (value > 1.0)? 1.0: value;
  return acos(value) * 180.0 / M_PI;
}

double Sim3Test::TranslationError(Vec3 t1, Vec3 t2) {
  return (t1 - t2).norm() / t1.norm() * 100;
}

double Sim3Test::ScaleError(double s1, double s2) {
  return fabs(s2 - s1) / s1 * 100;
}

double Sim3Test::HomogeneousError(Mat4 H1, Mat4 H2) {
  return (H1 - H2).norm() / H1.norm() * 100;
}

TEST(Lie, Sim3) {
  Sim3Test test;
  test.RunTest(100);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  srand(0);
  return RUN_ALL_TESTS();
}
