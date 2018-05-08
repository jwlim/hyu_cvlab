// similarity.cc
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include <time.h> 
#include <math.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/NonLinearOptimization>
#include <float.h> 
#include "rvslam_util.h"
#include "similarity.h"

namespace rvslam {

bool RobustEstimateSimilarity(const Mat3X& pts2d0,
                              const Mat3X& pts2d1,
                              const Mat3X& pts3d0,
                              const Mat3X& pts3d1,
                              Mat34* best_model,
                              std::vector<int>* best_inliers,
                              double *best_cost,
                              double *scale,
                              double inlier_threshold,
                              double failure_probability,
                              int max_iterations) {
  CHECK_NOTNULL(best_model);
  CHECK_LT(failure_probability, 1.0);
  CHECK_GT(failure_probability, 0.0);
  CHECK_EQ(pts2d0.cols(), pts2d1.cols());
  CHECK_EQ(pts3d0.cols(), pts3d1.cols());
  CHECK_EQ(pts2d0.cols(), pts3d0.cols());
  
  //srand(time(NULL));

  Similarity sim(3, inlier_threshold, failure_probability, max_iterations);
  sim.RobustEstimatePose(pts2d0, pts2d1, pts3d0, pts3d1, best_model,
                         best_inliers, best_cost);
  *scale = sim.GetScale();
}

bool Similarity::RobustEstimatePose(const Mat3X& pts2d0, const Mat3X& pts2d1,
                                    const Mat3X& pts3d0, const Mat3X& pts3d1,
                                    Mat34* best_model, 
                                    std::vector<int>* best_inliers, 
                                    double *best_cost) {
  int count = pts2d0.cols(), good_count, max_good_count = 0;
  int i, iter, niters = std::max(max_iterations_, 1);
  std::vector<int>::iterator it;
  Mat3X spts2d0 = Mat::Ones(3, model_points_);
  Mat3X spts2d1 = Mat::Ones(3, model_points_);
  Mat3X spts3d0 = Mat::Zero(3, model_points_);
  Mat3X spts3d1 = Mat::Zero(3, model_points_);
  Mat1X err0(count), err1(count); 
  Mask mask(count), temp_mask(count), mask0(count), mask1(count);
  Mat34 sRt, temp_sRt;

  if (count == model_points_) {
    mask.fill(true);
    max_good_count = model_points_;
    ComputeSimilarity(pts3d0, pts3d1, &sRt);
  }
  else {
    for (iter = 0; iter < niters; ++iter) {
      bool found = GetSubset3D3D(pts2d0, pts2d1, pts3d0, pts3d1,
                                 &spts2d0, &spts2d1, &spts3d0, &spts3d1,
                                 1000);
      if (!found) {
        if (iter == 0)
          return false;
        break;
      }
      ComputeSimilarity(spts3d0, spts3d1, &temp_sRt);
      FindInliers(pts3d0, pts3d1, temp_sRt, &err0, &mask0);
      FindInliers(pts3d1, pts3d0, InverseSimilarity(temp_sRt), &err1, &mask1);
      temp_mask = BitwiseAnd(mask0, mask1);
      good_count = temp_mask.count();
      if (good_count > std::max(max_good_count, model_points_ - 1)) {
        sRt = temp_sRt;
        mask = temp_mask;
        max_good_count = good_count;
        niters = UpdateRanSaCNumIters((double)(good_count)/count, niters);
      }
    }

    if (max_good_count < model_points_ + 1) {
      best_model->setIdentity(3, 4);
      if (best_cost) *best_cost = 1e10; 
      if (best_inliers) best_inliers->clear();
      return false;
    }
    // Refinement
    spts3d0 = Mat::Zero(3, max_good_count); 
    spts3d1 = Mat::Zero(3, max_good_count);
    bool* mask_ptr = mask.data();
    int n = 0;
    for (i = 0; i < count; ++i) {
      if (*(mask_ptr++) == true) {
        spts3d0.block(0, n, 3, 1) = pts3d0.block(0, i, 3, 1);
        spts3d1.block(0, n++, 3, 1) = pts3d1.block(0, i, 3, 1);
      }
    }
    ComputeSimilarity(spts3d0, spts3d1, &sRt);
  }

  *best_model = sRt;
  if (best_cost) {
    ComputeError(pts3d0, pts3d1, sRt, &err0);
    ComputeError(pts3d1, pts3d0, InverseSimilarity(sRt), &err1);
    *best_cost = err0.sum() + err1.sum();
  }
  if (best_inliers) {
    best_inliers->clear();
    best_inliers->resize(max_good_count);
    it = best_inliers->begin();
    bool* mask_ptr = mask.data();
    for (i = 0; i < count; ++i)
      if (*(mask_ptr++) == true)
        *(it++) = i;  
  }

  return true;
}

void Similarity::ComputeSimilarity(const Mat3X& pts0, const Mat3X& pts1,
                                   Mat34* solution) {
  // Arun, K. Somani, Thomas S. Huang, and Steven D. Blostein. "Least-squares
  // fitting of two 3-D point sets." Pattern Analysis and Machine Intelligence,
  // IEEE Transactions on 5 (1987): 698-700.

  // Horn, Berthold KP, Hugh M. Hilden, and Shahriar Negahdaripour.
  // "Closed-form solution of absolute orientation using orthonormal matrices."
  // JOSA A 5.7 (1988): 1127-1135.

  int n = pts0.cols();
  double n_inv = 1. / n;

  // ==== Stage 1: Compute centroids
  Vec3 c0 = pts0.rowwise().sum() * n_inv;
  Vec3 c1 = pts1.rowwise().sum() * n_inv;

  // ==== Stage 2: Subtract centroids from pointsi
  Mat3X spts0 = pts0.colwise() - c0;
  Mat3X spts1 = pts1.colwise() - c1;

  // ==== Stage 3: Singular value decomposition
  Mat3 H = Mat3::Zero();
  for (int i = 0; i < n; ++i)
    H += spts0.col(i) * spts1.col(i).transpose();
  Eigen::JacobiSVD<Mat> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // ==== Stage 4: Compute rotation matrix
  Mat3 R = svd.matrixV() * svd.matrixU().transpose();

  // ==== Stage 5: Compute scale
  double s = spts1.norm() / spts0.norm();
  scale_ = s;

  // ==== Stage 6: Compute translation vector
  Vec3 t = c1 - s * R * c0;

  *solution << s * R, t;
}

inline Mat34 Similarity::InverseSimilarity(const Mat34& sRt) {
  Mat4 T = Mat4::Identity();
  T.block(0, 0, 3, 4) = sRt;
  return T.inverse().block(0, 0, 3, 4);
}

void Similarity::ComputeError(const Mat3X& pts0, const Mat3X& pts1,
                              const Mat& model, Mat1X* err) {
  int i, count = err->cols();
  double t = inlier_threshold_ * inlier_threshold_;
  const double* sRt = model.data();
  for (i = 0; i < count; ++i) {
    double dx =   sRt[0] * pts0(0, i) + sRt[3] * pts0(1, i)
                + sRt[6] * pts0(2, i) + sRt[9] - pts1(0, i);
    double dy =   sRt[1] * pts0(0, i) + sRt[4] * pts0(1, i)
                + sRt[7] * pts0(2, i) + sRt[10] - pts1(1, i);
    double dz =   sRt[2] * pts0(0, i) + sRt[5] * pts0(1, i)
                + sRt[8] * pts0(2, i) + sRt[11] - pts1(2, i);
    double e = dx * dx + dy * dy + dz * dz;
    (*err)(i) = (e > t)? t: e;
  }
}

}  // namespace rvslam
