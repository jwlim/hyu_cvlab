// triangulation.cc
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//
// TriangulatePoints from OpenCV3.0.0

#include "rvslam_common.h"
#include "triangulation.h"

namespace rvslam {

void TriangulatePoints(const Mat34& pose0, const Mat34& pose1, 
                       const Mat3X& pts0, const Mat3X& pts1, 
                       Eigen::ArrayXXd* pts_4d) {
  int n = pts0.cols();
  Mat A(4, 4), U(4, 4), W(4, 1), V(4, 4);
  const Mat3X* pts[2] = { &pts0, &pts1 };
  const Mat34* pose[2] = { &pose0, &pose1 };
  
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 2; ++j) {
      double x, y;
      x = (*(pts[j]))(0, i);
      y = (*(pts[j]))(1, i);
      for (int k = 0; k < 4; ++k) {
        A(j*2+0, k) = x * (*pose[j])(2,k) - (*pose[j])(0,k);
        A(j*2+1, k) = y * (*pose[j])(2,k) - (*pose[j])(1,k);
      }
    }
    Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
    pts_4d->col(i) = svd.matrixV().col(3);
  }
  pts_4d->rowwise() *= 1. / pts_4d->row(3);
}

}
