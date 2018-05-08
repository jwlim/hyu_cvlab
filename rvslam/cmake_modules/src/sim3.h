// sim3.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_SIM3_H_
#define _RVSLAM_SIM3_H_

#include <cmath>
#include <Eigen/Geometry>
#include "rvslam_util.h"

namespace rvslam {

class Sim3 {
    
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sim3() {
    r_.setIdentity();
    t_.fill(0.);
    s_ = 1.;
  }

  Sim3(const Sim3& S)
      : r_(S.Rotation()), t_(S.Translation()), s_(S.Scale()) {}

  Sim3(const Eigen::Quaterniond& r, const Vec3& t, double s)
      : r_(r), t_(t), s_(s) {
    r_.normalize();
  }
  
  Sim3(const Mat3& R, const Vec3& t, double s)
    : r_(Eigen::Quaterniond(R)), t_(t), s_(s) {}

  Sim3(const Mat34& T) {
    Mat3 sR = GetRot(T);
    s_ = pow(sR.determinant(), 1./3.);
    r_ = Eigen::Quaterniond(sR / s_);
    t_ = GetTr(T);
  }

  Sim3(const Mat4& H) {
    Mat34 T = ToMat34(H); 
    Mat3 sR = GetRot(T);
    s_ = pow(sR.determinant(), 1./3.);
    r_ = Eigen::Quaterniond(sR / s_);
    t_ = GetTr(T);
  }
  
  Sim3(const double* spose) {
    const double& a0 = spose[0];
    const double& a1 = spose[1];
    const double& a2 = spose[2];
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
    r_ = Eigen::Quaterniond(q0, q1, q2, q3);
    t_[0] = spose[3]; 
    t_[1] = spose[4]; 
    t_[2] = spose[5]; 
    s_ = spose[6]; 
  }

  // From sim3 (Lie algebra) to Sim3 (Lie group)
  Sim3(const Vec7 & u) {
    Vec3 v, w;
    for (int i = 0; i < 3; ++i) {
      v[i] = u[i];
      w[i] = u[i + 3];
    }
    double lambda = u[6];
    double lambda2 = lambda * lambda;
    double a = w.norm();
    double a2 = a * a;
    double sa = sin(a);
    double ca = cos(a);
    Mat3 W = CrossProductMatrix(w);
    s_ = exp(lambda);
    Mat3 W2 = W * W;
    Mat3 I = Mat3::Identity();
    Mat3 R;

    double eps = 1e-5;
    double c0, c1, c2;
    if (a < eps) {
      R = I + W + 0.5 * W2;
      if (fabs(lambda) < eps)
        c0 = s_;
      else
        c0 = (s_ - 1) / lambda;
      c1 = (3 * s_ * ca - lambda * s_ * ca - a * s_ * sa) / 6;
      c2 = s_ * a / 6 - lambda * s_ * ca / 24;
    }
    else {
      R = I +  W * sa / a + W2 * (1 - ca) / a2;
      if (fabs(lambda) < eps) {
        c0 =   s_;
        c1 =   (2 * s_ * sa - a * s_ * ca + lambda * s_ * sa) / (2 * a);
        c2 =   s_ / a2 - s_ * sa / (2 * a) 
             - (2 * s_ * ca + lambda * s_ * ca) / (2 * a2);
      }
      else {
        c0 =   (s_ - 1) / lambda;
        c1 =   (a * (1 - s_ * ca) + s_ * lambda * sa) / (a * (lambda2 + a2));
        c2 =   (s_ - 1) / (lambda * a2) - (s_ * sa) / (a * (lambda2 + a2)) 
             - (lambda * (s_ * ca - 1)) / (a2 * (lambda2 + a2));
      }
    }
    Mat3 V = c0 * I + c1 * W + c2 * W2;
    t_ = V * v;
    r_ = Eigen::Quaterniond(R);
  }

  Vec7 Log() const {
    Vec7 u;
    double lambda = log(s_);
    double lambda2 = lambda * lambda;
    Mat3 R = r_.toRotationMatrix();
    Vec3 w = MatLog(R);
    Mat3 W = CrossProductMatrix(w);
    Mat3 W2 = W * W;
    double a = w.norm();
    double a2 = a * a;
    double sa = sin(a);
    double ca = cos(a);
    Mat3 I = Mat3::Identity();
    
    double eps = 1e-5;
    double c0, c1, c2;
    if (a < eps) {
      R = I + W + 0.5 * W2;
      if (fabs(lambda) < eps)
        c0 = s_;
      else
        c0 = (s_ - 1) / lambda;
      c1 = (3 * s_ * ca - lambda * s_ * ca - a * s_ * sa) / 6;
      c2 = s_ * a / 6 - lambda * s_ * ca / 24;
    }
    else {
      R = I +  W * sa / a + W2 * (1 - ca) / a2;
      if (fabs(lambda) < eps) {
        c0 =   s_;
        c1 =   (2 * s_ * sa - a * s_ * ca + lambda * s_ * sa) / (2 * a);
        c2 =   s_ / a2 - s_ * sa / (2 * a) 
             - (2 * s_ * ca + lambda * s_ * ca) / (2 * a2);
      }
      else {
        c0 =   (s_ - 1) / lambda;
        c1 =   (a * (1 - s_ * ca) + s_ * lambda * sa) / (a * (lambda2 + a2));
        c2 =   (s_ - 1) / (lambda * a2) - (s_ * sa) / (a * (lambda2 + a2)) 
             - (lambda * (s_ * ca - 1)) / (a2 * (lambda2 + a2));
      }
    }
    Mat3 V = c0 * I + c1 * W + c2 * W2;
    Vec3 v = V.lu().solve(t_);

    for (int i = 0; i < 3; ++i) {
      u[i] = v[i];
      u[i + 3] = w[i];
    }
    u[6] = lambda;
    return u;
  }

  Mat7 Adj() const {
    Mat7 A = Mat::Zero(7, 7);
    Mat3 R = r_.toRotationMatrix(); 
    Mat3 Tx = CrossProductMatrix(t_);
    A.block(0, 0, 3, 3) = s_ * R;
    A.block(0, 3, 3, 3) = Tx * R;
    A.block(0, 6, 3, 1) = -t_;
    A.block(3, 3, 3, 3) = R;
    A(6, 6) = 1;
    return A;
  }

  void GetSPose(double* spose) const {
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
    spose[0] = r[0];
    spose[1] = r[1];
    spose[2] = r[2];
    spose[3] = t_[0];
    spose[4] = t_[1];
    spose[5] = t_[2];
    spose[6] = s_;
  }
  
  void GetLog(double* u) const {
    double lambda = log(s_);
    double lambda2 = lambda * lambda;
    Mat3 R = r_.toRotationMatrix();
    Vec3 w = MatLog(R);
    Mat3 W = CrossProductMatrix(w);
    Mat3 W2 = W * W;
    double a = w.norm();
    double a2 = a * a;
    double sa = sin(a);
    double ca = cos(a);
    Mat3 I = Mat3::Identity();
    
    double eps = 1e-5;
    double c0, c1, c2;
    if (a < eps) {
      R = I + W + 0.5 * W2;
      if (fabs(lambda) < eps)
        c0 = s_;
      else
        c0 = (s_ - 1) / lambda;
      c1 = (3 * s_ * ca - lambda * s_ * ca - a * s_ * sa) / 6;
      c2 = s_ * a / 6 - lambda * s_ * ca / 24;
    }
    else {
      R = I +  W * sa / a + W2 * (1 - ca) / a2;
      if (fabs(lambda) < eps) {
        c0 =   s_;
        c1 =   (2 * s_ * sa - a * s_ * ca + lambda * s_ * sa) / (2 * a);
        c2 =   s_ / a2 - s_ * sa / (2 * a) 
             - (2 * s_ * ca + lambda * s_ * ca) / (2 * a2);
      }
      else {
        c0 =   (s_ - 1) / lambda;
        c1 =   (a * (1 - s_ * ca) + s_ * lambda * sa) / (a * (lambda2 + a2));
        c2 =   (s_ - 1) / (lambda * a2) - (s_ * sa) / (a * (lambda2 + a2)) 
             - (lambda * (s_ * ca - 1)) / (a2 * (lambda2 + a2));
      }
    }
    Mat3 V = c0 * I + c1 * W + c2 * W2;
    Vec3 v = V.lu().solve(t_);

    for (int i = 0; i < 3; ++i) {
      u[i] = v[i];
      u[i + 3] = w[i];
    }
    u[6] = lambda;
  }

  Sim3 Inverse() const {
    return Sim3(r_.conjugate(), r_.conjugate() * ((-1. / s_) * t_), 1. / s_);
  }

  Vec3 Transform(const Vec3& v) {
    return s_ * (r_ * v)  + t_;
  }

  Sim3 operator*(const Sim3& u) const {
    Sim3 result;
    result.r_ = r_ * u.r_;
    result.t_ = s_ * (r_ * u.t_) +t_;
    result.s_ = s_ * u.s_;
    return result;
  }

  Sim3& operator*=(const Sim3& u){
    Sim3 result = (*this) * u;
    *this = result;
    return *this;
  }

  Vec3 operator*(const Vec3& v) {
    return  s_ * (r_ * v) + t_;
  }

  Mat34 Transformation() const {
    Mat34 T;
    T.block(0, 0, 3, 3) = s_ * r_.toRotationMatrix();
    T.block(0, 3, 3, 1) = t_;
    return T;
  }

  Mat4 HomogeneousMat() const {
    return ToMat4(Transformation()); 
  }

  void SetRotation(const Eigen::Quaterniond& r) {r_ = r;}
  
  void SetTranslation(const Vec3& t) {t_ = t;}

  void SetScale(double s) {s_ = s;}
  
  Eigen::Quaterniond Rotation() const {return r_;}

  Vec3 Translation() const {return t_;}

  double Scale() const {return s_;}

 protected:
  Eigen::Quaterniond r_;
  Vec3 t_;
  double s_;
};

inline std::ostream& operator <<(std::ostream& os, const Sim3& sim3) {
  os << sim3.Transformation() << std::endl;
  return os;
}

}  // namespace rvslam

#endif  // _RVSLAM_SIM3_H_
