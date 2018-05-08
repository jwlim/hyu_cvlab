// estimate_mat.cc
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
#include "estimate_rot.h"

namespace rvslam {

bool RobustEstimateRotationMat(const Mat3X& pts0,
                                    const Mat3X& pts1,
                                    Mat34* best_model,
                                    std::vector<int>* best_inliers,
                                    double *best_cost,
                                    double inlier_threshold,
                                    double failure_probability,
                                    int max_iterations) {
  CHECK_NOTNULL(best_model);
  CHECK_LT(failure_probability, 1.0);
  CHECK_GT(failure_probability, 0.0);
  CHECK_EQ(pts0.cols(), pts1.cols());
  
  //srand(time(NULL));

  EstimateRot rot(2, inlier_threshold, failure_probability, max_iterations);
  rot.RobustEstimatePose(pts0, pts1, best_model, best_inliers, best_cost);
}

bool EstimateRot::RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
                                    Mat34* best_model, 
                                    std::vector<int>* best_inliers, 
                                    double *best_cost) {
  int count = pts0.cols(), good_count, max_good_count = 0;
  int i, iter, niters = std::max(max_iterations_, 1);
  std::vector<int>::iterator it;
  Mat3X spts0 = Mat::Ones(3, model_points_); 
  Mat3X spts1 = Mat::Ones(3, model_points_);
  Mat1X err(count); 
  Mask mask(count), temp_mask(count);
  Mat3 rot, temp_rot;

  if (count == model_points_) {
    mask.fill(true);
    max_good_count = model_points_;
    ComputeRotationMat(pts0, pts1, &rot);
  }
  else {
    for (iter = 0; iter < niters; ++iter) {
      bool found = GetSubset2D2D(pts0, pts1, &spts0, &spts1, 1000);
      if (!found) {
        if (iter == 0)
          return false;
        break;
      }
      ComputeRotationMat(spts0, spts1, &temp_rot);
      good_count = FindInliers(pts0, pts1, temp_rot, &err, &temp_mask);
      if (good_count > std::max(max_good_count, model_points_ - 1)) {
        rot = temp_rot;
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
    RotFunctorNumericalDiff functor;
    functor.pts0_ = Mat::Ones(3, max_good_count); 
    functor.pts1_ = Mat::Ones(3, max_good_count);
    bool* mask_ptr = mask.data();
    int n = 0;
    for (i = 0; i < count; ++i) {
      if (*(mask_ptr++) == true) {
        functor.pts0_.block(0, n, 2, 1) = pts0.block(0, i, 2, 1);
        functor.pts1_.block(0, n++, 2, 1) = pts1.block(0, i, 2, 1);
      }
    }
    Vec r = MatLog(rot);
    Eigen::LevenbergMarquardt<RotFunctorNumericalDiff> lm(functor);
    lm.parameters.maxfev = pts0.cols() * 2;
    lm.parameters.xtol = 1e-12;
    lm.minimize(r);
    rot = ExpMap(r);
  }

  *best_model << rot, Vec3(0, 0, 0);
  if (best_cost) {
    ComputeError(pts0, pts1, rot, &err);
    *best_cost = err.sum();
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

void EstimateRot::ComputeRotationMat(const Mat3X& pts0, const Mat3X& pts1,
                                     Mat3* solution) {
  // Wahba's problem, use solution by SVD
  // https://en.wikipedia.org/wiki/Wahba%27s_problem
  Mat3X npts0 = pts0.array().rowwise() * (1. / pts0.colwise().norm().array()); 
  Mat3X npts1 = pts1.array().rowwise() * (1. / pts1.colwise().norm().array()); 
  Mat3 B = npts1 * npts0.transpose();
  Eigen::JacobiSVD<Mat3> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 U = svd.matrixU();
  Mat3 Vt = svd.matrixV().transpose();
  Mat3 M;
  M << 1, 0, 0, 0, 1, 0, 0, 0, U.determinant()*Vt.determinant();
  *solution = U * M * Vt;
}

void EstimateRot::ComputeError(const Mat3X& pts0, const Mat3X& pts1,
                               const Mat& rot, Mat1X* err) {
  int i, count = err->cols();
  double t = inlier_threshold_ * inlier_threshold_;
  const double* R = rot.data();
  for (i = 0; i < count; ++i) {
    double w = 1 / (R[2] * pts0(0, i) + R[5] * pts0(1, i) + R[8]);
    double dx = (R[0] * pts0(0, i) + R[3] * pts0(1, i) + R[6]) * w - pts1(0, i);
    double dy = (R[1] * pts0(0, i) + R[4] * pts0(1, i) + R[7]) * w - pts1(1, i);
    double e = dx * dx + dy * dy;
    (*err)(i) = (e > t)? t: e;
  }
}

}  // namespace rvslam
