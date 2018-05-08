// rvslam_util.h
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_RVSLAM_UTIL_H_
#define _RVSLAM_RVSLAM_UTIL_H_

#include <math.h>
#include <map>
#include <iostream>
#include <vector>
#include <glog/logging.h>

#include "rvslam_common.h"

namespace rvslam {

inline Mat34 IdentityTransform() {
  Mat34 P = Mat4::Identity().block(0, 0, 3, 4);
  return P;
}

template <typename T, int d, int n> inline
Eigen::Matrix<T, d, n> UnitVector(const Eigen::Matrix<T, d, n>& v) {
  return v / (v.norm() + 1e-9);
}

template <typename T, int d, int n> inline
Eigen::Matrix<T, d + 1, n> Hom(const Eigen::Matrix<T, d, n>& x) {
  Eigen::Matrix<T, d + 1, n> h(x.rows() + 1, x.cols());
  h.block(0, 0, x.rows(), x.cols()) = x;
  h.row(x.rows()).fill(1.0);
  return h;
}

template <typename T, int d> inline
Eigen::Matrix<T, d + 1, 1> Hom(const Eigen::Matrix<T, d, 1>& x) {
  Eigen::Matrix<T, d + 1, 1> h;
  h << x, 1.0;
  return h;
}

template <typename T, int d, int n> inline
Eigen::Matrix<T, d - 1, n> Euc(const Eigen::Matrix<T, d, n>& h) {
  Eigen::Array<T, d - 1, n> x = h.block(0, 0, h.rows() - 1, h.cols()).array();
  Eigen::Array<T, 1, n> y = h.row(h.rows() - 1);
  x.rowwise() /= y;
  return x.matrix();
}

template <typename T, int d> inline
Eigen::Matrix<T, d - 1, 1> Euc(const Eigen::Matrix<T, d, 1>& h) {
  Eigen::Matrix<T, d - 1, 1> x = h.block(0, 0, d - 1, 1) / h(d - 1, 0);
  return x;
}

inline Mat3X Project(const Mat34& P, const Mat3X& X) {
  Mat3X x = P * Hom(X);
  return (x.array().rowwise() / x.row(2).array()).matrix();
}

//template <typename T> inline
//T Project(const Eigen::MatrixBase<T>& X) {
//  return (X.array().rowwise() / X.row(X.rows() - 1).array()).matrix();
//}

//inline Vec3 Project(const Mat34& P, const Vec3& X) {
//  Vec3 x = P * Hom(X);
//  return x / x(2);
//}

inline Vec2 ProjectEuc(const Mat34& P, const Vec3& X) {
  Vec3 x = P * Hom(X);
  return Euc(x);
}

inline bool Project(const Mat34& P, const Vec3& X, double z_thr, Vec3* x) {
  Vec3 p = P * Hom(X);
  if (p(2) <= z_thr) return false;
  *x = p / p(2);
  return true;
}

inline bool Project(const Mat34& P, const Vec3& X, Vec3* x) {
  return Project(P, X, 0.0, x);
}

inline Mat4 ToMat4(const Mat34& P) {
  Mat4 H;
  H << P, 0.0, 0.0, 0.0, 1.0;
  return H;
}

inline Mat34 ToMat34(const Mat4& H) {
  Mat34 P = H.block(0, 0, 3, 4);
  return P;
}

inline Mat3 GetRot(const Mat34& P) { return P.block(0, 0, 3, 3); }
inline Vec3 GetTr(const Mat34& P) { return P.block(0, 3, 3, 1); }
inline void SetRot(const Mat3& R, Mat34* P) { P->block(0, 0, 3, 3) = R; }
inline void SetTr(const Vec3& tr, Mat34* P) { P->block(0, 3, 3, 1) = tr; }

inline Vec3 GetRotVec(const Vec6& pose) { return pose.block(0, 0, 3, 1); }
inline Vec3 GetTr(const Vec6& pose) { return pose.block(3, 0, 3, 1); }
inline void SetRotVec(const Vec3& rot, Vec6* p) { p->block(0, 0, 3, 1) = rot; }
inline void SetTr(const Vec3& tr, Vec6* p) { p->block(3, 0, 3, 1) = tr; }

inline Mat34 InverseTransform(const Mat34& P, bool rigid = false) {
  return ToMat34(ToMat4(P).inverse());
}

inline Mat34 MergedTransform(const Mat34& P1, const Mat34& P2) {
  return P1 * ToMat4(P2);
}

inline Mat34 RelativeTransform(const Mat34& P_ref, const Mat34& P) {
  return P * ToMat4(InverseTransform(P_ref));
//  return InverseTransform(P_ref) * ToMat4(P);
}

inline Vec3 RelativePosition(const Mat34& P_ref, const Vec3& X) {
  return InverseTransform(P_ref) * Hom(X);
}

inline Mat3 CrossProductMatrix(const Vec3& x) {
  Mat3 X;
  X <<     0, -x(2),  x(1),
        x(2),     0, -x(0),
       -x(1),  x(0),     0;
  return X;
}

inline Mat3 RotationRodrigues(const Vec3& axis) {
  double theta = axis.norm();
  Vec3 w = axis / theta;
  Mat3 W = CrossProductMatrix(w);
  return Mat3::Identity() + sin(theta) * W + (1 - cos(theta)) * W * W;
}

inline Mat3 ExpMap(const Vec3& w) {
  const double eps = 1e-5;
  double a = w.norm();
  Mat3 W = CrossProductMatrix(w);
  Mat3 I = Mat3::Identity();
  if (a < eps) {
    return I + W + 0.2 * W * W;
  } else {
    return I + W * (sin(a) / a) + W * W * ((1 - cos(a)) / a / a);
  }
}

inline Vec3 MatLog(const Mat3& R) {
  const double eps = 1e-5;
  double a = acos((R.trace() - 1) / 2);
  Vec3 w(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  if (a < eps) {
    w *= 0.5;
  } else if (a < M_PI - eps) {
    w *= a / 2 / sin(a);
  } else {
    Mat3 S = 0.5 * (R - Mat3::Identity());
    double b = sqrt(S(0, 0) + 1);
    double c = sqrt(S(1, 1) + 1);
    double d = sqrt(S(2, 2) + 1);
    if (b > eps) {
      c = S(1, 0) / b;
      d = S(2, 0) / b;
    } else if (c > eps) {
      b = S(0, 1) / c;
      d = S(2, 1) / c;
    } else {
      b = S(0, 2) / d;
      c = S(1, 2) / d;
    }
    w(0) = b;
    w(1) = c;
    w(2) = d;
  }
  return w; 
}

inline Mat34 MakeTransform(const Vec6& param) {
  Mat34 P;
  P << RotationRodrigues(param.segment(0, 3)), param.segment(3, 3);
  return P;
}

//-----------------------------------------------------------------------------

template <typename T> inline
int FindInSortedArray(const std::vector<T>& array, const T& val) {
  typename std::vector<T>::const_iterator it =
      lower_bound(array.begin(), array.end(), val);
  return (it == array.end() || *it != val) ? -1 :
      static_cast<int>(it - array.begin());
}

template <typename K, typename V> inline
const V& FindInMapOrDie(const std::map<K, V>& the_map, const K& key) {
  typename std::map<K, V>::const_iterator it = the_map.find(key);
  CHECK(it != the_map.end());
  return it->second;
}

template <typename K, typename V> inline
V& FindInMapOrDie(std::map<K, V>& the_map, const K& key) {
  typename std::map<K, V>::iterator it = the_map.find(key);
  CHECK(it != the_map.end());
  return it->second;
}

template <typename K, typename V> inline
const V& FindInMap(const std::map<K, V>& the_map, const K& key, const V& def) {
  typename std::map<K, V>::const_iterator it = the_map.find(key);
  return it == the_map.end() ? def : it->second;
}

template <typename K, typename V> inline
V& FindInMap(std::map<K, V>& the_map, const K& key, V& def) {
  typename std::map<K, V>::iterator it = the_map.find(key);
  return it == the_map.end() ? def : it->second;
}

template <typename K, typename V> inline
bool FindInMap(const std::map<K, V>& the_map, const K& key, const V* value) {
  typename std::map<K, V>::const_iterator it = the_map.find(key);
  if (it == the_map.end()) return false;
  if (value != NULL) *value = it->second;
  return true;
}

template <typename K, typename V> inline
bool FindInMap(std::map<K, V>& the_map, const K& key, V* value) {
  typename std::map<K, V>::iterator it = the_map.find(key);
  if (it == the_map.end()) return false;
  if (value != NULL) *value = it->second;
  return true;
}

//-----------------------------------------------------------------------------

inline size_t SplitStr(const std::string& str, const std::string& delim,
                       std::vector<std::string>* tokens) {
  tokens->clear();
  size_t i, j, npos = std::string::npos;
  for (i = 0; (i = str.find_first_not_of(delim, i)) != npos; i = j) {
    j = str.find_first_of(delim, i);
    if (j == npos) tokens->push_back(str.substr(i));
    else tokens->push_back(str.substr(i, j - i));
  }
  return tokens->size();
}

//-----------------------------------------------------------------------------

template <typename T> inline
std::ostream& operator<<(std::ostream& os, std::vector<T>& array) {
  os << "[";
  if (!array.empty()) os << array[0];
  for (int i = 1; i < array.size(); ++i) os << ", " << array[i];
  os << "]";
  return os;
}

//-----------------------------------------------------------------------------

template <typename T1, int d1, int n1, typename T2, int d2, int n2> inline
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> BitwiseAnd(
    const Eigen::Array<T1, d1, n1>& A1, const Eigen::Array<T2, d2, n2>& A2) {
  int r = A1.rows();
  int c = A1.cols();
  int t = r * c;
  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> A(r, c);
  bool* a = A.data();
  const T1* a1 = A1.data();
  const T2* a2 = A2.data();
  for (int i = 0; i < t; ++i) {
    T1 x = *(a1++);
    T2 y = *(a2++);
    *(a++) = (x == 0 || y == 0) ? 0 : 1;
  }
  return A;
}

}  // namespace rvslam
#endif  // _RVSLAM_RVSLAM_UTIL_H_
