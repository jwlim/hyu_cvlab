// ransac.c
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//
// Most of the functions are modified from OpenCV3.0.0

#include <cstdlib>
#include <algorithm>
#include <vector>
#include <float.h> 
#include "ransac.h"

namespace rvslam {

bool RanSaC::CheckSubset(const Mat3X& spts) {
  const float THRESHOLD = 0.996f;
  int i, j, k = model_points_ - 1;

  // Make sure the last selected point is not on the line connecting two 
  // previously selected points
  for (i = 0; i < k; ++i) {
    Vec3 d1 = spts.col(i) - spts.col(k);
    float n1 = d1(0) * d1(0) + d1(1) * d1(1);
    for (j = 0; j < i; ++j) {
      Vec3 d2 = spts.col(j) - spts.col(k);
      float denom = (d2(0) * d2(0) + d2(1) * d2(1)) * n1;
      float num = d1(0) * d2(0) + d1(1) * d2(1);
      if (num * num > THRESHOLD * THRESHOLD * denom)
        return false;
    }
  }
  return true;
}

int RanSaC::FindInliers(const Mat3X& pts0, const Mat3X& pts1,
                            const Mat& model, Mat1X* err, Mask* mask) {
  ComputeError(pts0, pts1, model, err);
  double t = inlier_threshold_ * inlier_threshold_;
  int i, n = (int)err->size(), nz = 0;
  bool* maskptr = mask->data();
  double* errptr = err->data();
  for (i = 0; i < n; i++)
    *(maskptr++) = (*(errptr++) < t);
  return mask->count();
}

int RanSaC::UpdateRanSaCNumIters(double p_inlier, int niters) {
  double num = failure_probability_;
  double denom = 1. - std::pow(p_inlier, model_points_);
  if (denom < DBL_MIN)
      return 0;

  num = std::log(num);
  denom = std::log(denom);

  return -num >= niters * (-denom) ? niters: round(num / denom);
}

bool RanSaC::GetSubset2D2D(const Mat3X& pts0, const Mat3X& pts1,
                           Mat3X* spts0, Mat3X* spts1,
                           int max_attempts) {
  Mat1X idx(model_points_);
  int i, j, iters = 0;
  int count = pts0.cols();
 
  for (; iters < max_attempts; ++iters) {
    for (i = 0; i < model_points_ && iters < max_attempts;) {
      int idx_i = 0;
      for (;;) {
        idx_i = idx(i) = rand() % count;
        for (j = 0; j < i; j++)
          if (idx_i == idx(j))
            break;
        if (j == i)
          break;
      }
      spts0->block(0, i, 2, 1) = pts0.block(0, idx_i, 2, 1);
      spts1->block(0, i, 2, 1) = pts1.block(0, idx_i, 2, 1);
      i++;
    }
    if (i == model_points_ && !(CheckSubset(*spts0) && CheckSubset(*spts1)))
      continue;
    break;
  }
  return i == model_points_ && iters < max_attempts;
}

bool RanSaC::GetSubset2D3D(const Mat3X& pts2d, const Mat3X& pts3d,
                           Mat3X* spts2d, Mat3X* spts3d,
                           int max_attempts) {
  Mat1X idx(model_points_);
  int i, j, iters = 0;
  int count = pts2d.cols();
 
  for (; iters < max_attempts; ++iters) {
    for (i = 0; i < model_points_ && iters < max_attempts;) {
      int idx_i = 0;
      for (;;) {
        idx_i = idx(i) = rand() % count;
        for (j = 0; j < i; j++)
          if (idx_i == idx(j))
            break;
        if (j == i)
          break;
      }
      spts2d->block(0, i, 2, 1) = pts2d.block(0, idx_i, 2, 1);
      spts3d->block(0, i, 3, 1) = pts3d.block(0, idx_i, 3, 1);
      i++;
    }
    if (i == model_points_ && !CheckSubset(*spts2d))
      continue;
    break;
  }
  return i == model_points_ && iters < max_attempts;
}

bool RanSaC::GetSubset3D3D(const Mat3X& pts2d0, const Mat3X& pts2d1,
                           const Mat3X& pts3d0, const Mat3X& pts3d1,
                           Mat3X* spts2d0, Mat3X* spts2d1,
                           Mat3X* spts3d0, Mat3X* spts3d1,
                           int max_attempts) {
  Mat1X idx(model_points_);
  int i, j, iters = 0;
  int count = pts2d0.cols();
 
  for (; iters < max_attempts; ++iters) {
    for (i = 0; i < model_points_ && iters < max_attempts;) {
      int idx_i = 0;
      for (;;) {
        idx_i = idx(i) = rand() % count;
        for (j = 0; j < i; j++)
          if (idx_i == idx(j))
            break;
        if (j == i)
          break;
      }
      spts2d0->block(0, i, 2, 1) = pts2d0.block(0, idx_i, 2, 1);
      spts2d1->block(0, i, 2, 1) = pts2d1.block(0, idx_i, 2, 1);
      spts3d0->block(0, i, 3, 1) = pts3d0.block(0, idx_i, 3, 1);
      spts3d1->block(0, i, 3, 1) = pts3d1.block(0, idx_i, 3, 1);
      i++;
    }
    if (i == model_points_ && !(CheckSubset(*spts2d0) && CheckSubset(*spts2d1)))
      continue;
    break;
  }
  return i == model_points_ && iters < max_attempts;
}

}  // namespace rvslam
