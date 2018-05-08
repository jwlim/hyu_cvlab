// se3.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_SE3_H_
#define _RVSLAM_SE3_H_

#include <cmath>
#include <Eigen/Geometry>
#include "rvslam_util.h"

namespace rvslam {

class SE3 {
    
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3() {
    r_.setIdentity();
    t_.fill(0.);
  }
  
  SE3(const SE3& S)
      : r_(S.Rotation()), t_(S.Translation()) {}

  SE3(const Eigen::Quaterniond& r, const Vec3& t)
      : r_(r), t_(t) {
    r_.normalize();
  }
  
  SE3(const Mat3& R, const Vec3& t)
      : r_(Eigen::Quaterniond(R)), t_(t) {}

  SE3(const Mat34& T) {
    r_ = Eigen::Quaterniond((Mat3)T.block(0, 0, 3, 3));
    t_ = T.block(0, 3, 3, 1);
  }

  SE3(const Mat4& H) {
    r_ = Eigen::Quaterniond((Mat3)H.block(0, 0, 3, 3));
    t_ = H.block(0, 3, 3, 1);
  }
  
  SE3(const double* pose) {
    const double& a0 = pose[0];
    const double& a1 = pose[1];
    const double& a2 = pose[2];
    const double theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    double q0, q1, q2, q3;
    if (theta_squared > 1e-5) {
      const double theta = sqrt(theta_squared);
      const double half_theta = theta * 0.5;
      const double k = sin(half_theta) / theta;
      q0 = cos(half_theta);
      q1 = a0 * k;
      q2 = a1 * k;
      q3 = a2 * k;
    } else {
      const double k = 0.5;
      q0 = 1.0;
      q1 = a0 * k;
      q2 = a1 * k;
      q3 = a2 * k;
    }
    r_ = Eigen::Quaterniond(double(q0), double(q1), double(q2), double(q3));
    t_[0] = double(pose[3]); 
    t_[1] = double(pose[4]); 
    t_[2] = double(pose[5]); 
  }

  // From se3 (Lie algebra) to SE3 (Lie group)
  SE3(const Vec6 & u) {
    Vec3 v, w;
    for (int i = 0; i < 3; ++i) {
      v[i] = u[i];
      w[i] = u[i + 3];
    }
    double a = w.norm();
    double a2 = a * a;
    Mat3 W = CrossProductMatrix(w);
    Mat3 W2 = W * W;
    Mat3 I = Mat3::Identity();
    Mat3 R, V;

    double eps = 1e-5;
    double c1, c2, c3;
    if (a < eps) {
      R = I + W + 0.5 * W2;
      V = I + 0.5 * W + W2 / 6.;
    }
    else {
      c1 = sin(a) / a;
      c2 = (1 - cos(a)) / a2;
      c3 = (1 - c1) / a2;
      R = I + c1 * W + c2 * W2;
      V = I + c2 * W + c3 * W2;
    }
    t_ = V * v;
    r_ = Eigen::Quaterniond(R);
  }

  Vec6 Log() const {
    Vec6 u;
    Mat3 R = r_.toRotationMatrix();
    Vec3 w = MatLog(R);
    Mat3 W = CrossProductMatrix(w);
    Mat3 W2 = W * W;
    double a = w.norm();
    double a2 = a * a;
    Mat3 I = Mat3::Identity();
    Mat3 V, V_inv;
    Vec3 v;

    double eps = 1e-5;
    double c1, c2;
    if (a < eps) {
      V = I + 0.5 * W + W2;
      v = V.inverse() * t_;
    }
    else {
      c1 = sin(a) / a;
      c2 = (1 - cos(a)) / a2;
      V_inv = I - 0.5 * W + (1. / a2) * (1 - (c1 / 2. / c2)) * W2;
      v = V_inv * t_;
    }
    u << v, w;
    return u;
  }

  Mat6 Adj() const {
    Mat6 A = Mat::Zero(6, 6);
    Mat3 R = r_.toRotationMatrix(); 
    Mat3 Tx = CrossProductMatrix(t_);
    A.block(0, 0, 3, 3) = R;
    A.block(0, 3, 3, 3) = Tx * R;
    A.block(3, 3, 3, 3) = R;
    return A;
  }
  
  void GetPose(double* pose) const {
    Vec3 r;
    const double& q1 = r_.x();
    const double& q2 = r_.y();
    const double& q3 = r_.z();
    const double sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3;
    if (sin_squared_theta > 1e-5) {
      const double sin_theta = sqrt(sin_squared_theta);
      const double& cos_theta = r_.w();
      const double two_theta = 2.0 * ((cos_theta < 0.0) 
                                    ? atan2(-sin_theta, -cos_theta)
                                    : atan2(sin_theta, cos_theta));
      const double k = two_theta / sin_theta;
      r << q1 * k, q2 * k, q3 * k;
    } else {
      const double k = 2.0;
      r << q1 * k, q2 * k, q3 * k;
    }
    pose[0] = r[0];
    pose[1] = r[1];
    pose[2] = r[2];
    pose[3] = t_[0];
    pose[4] = t_[1];
    pose[5] = t_[2];
  }
  
  void GetLog(double* u) const {
    Mat3 R = r_.toRotationMatrix();
    Vec3 w = MatLog(R);
    Mat3 W = CrossProductMatrix(w);
    Mat3 W2 = W * W;
    double a = w.norm();
    double a2 = a * a;
    Mat3 I = Mat3::Identity();
    Mat3 V, V_inv;
    Vec3 v;

    double eps = 1e-5;
    double c1, c2;
    if (a < eps) {
      V = I + 0.5 * W + W2;
      v = V.inverse() * t_;
    }
    else {
      c1 = sin(a) / a;
      c2 = (1 - cos(a)) / a2;
      V_inv = I - 0.5 * W + (1. / a2) * (1 - (c1 / 2. / c2)) * W2;
      v = V_inv * t_;
    }

    for (int i = 0; i < 3; ++i) {
      u[i] = v[i];
      u[i + 3] = w[i];
    }
  }
  
  SE3 Inverse() const {
    return SE3(r_.conjugate(), r_.conjugate() * (-t_));
  }

  Vec3 Transform(const Vec3& v) {
    return r_ * v  + t_;
  }

  SE3 operator*(const SE3& u) const {
    SE3 result;
    result.r_ = r_ * u.r_;
    result.t_ = r_ * u.t_ +t_;
    return result;
  }

  SE3& operator*=(const SE3& u){
    SE3 result = (*this) * u;
    *this = result;
    return *this;
  }

  Vec3 operator*(const Vec3& v) {
    return  r_ * v + t_;
  }

  Mat34 Transformation() const {
    Mat34 T;
    T.block(0, 0, 3, 3) = r_.toRotationMatrix();
    T.block(0, 3, 3, 1) = t_;
    return T;
  }

  Mat4 HomogeneousMat() const {
    return ToMat4(Transformation()); 
  }

  void SetRotation(const Eigen::Quaterniond& r) {r_ = r;}
  
  void SetTranslation(const Vec3& t) {t_ = t;}

  Eigen::Quaterniond Rotation() const {return r_;}
  
  Vec3 Translation() const {return t_;}

 protected:
  Eigen::Quaterniond r_;
  Vec3 t_;
};

inline std::ostream& operator <<(std::ostream& os, const SE3& se3) {
  os << se3.Transformation() << std::endl;
  return os;
}

}  // namespace rvslam

#endif  // _RVSLAM_SE3_H_
