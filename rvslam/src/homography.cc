// homography.cc
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//
// Most of the functions are modified from OpenCV3.0.0

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
#include "homography.h"
#include "triangulation.h"

namespace rvslam {

bool RobustEstimateRelativePoseHomo(const Mat3X& pts0,
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

  Homography homo(4, inlier_threshold, failure_probability, max_iterations);
  homo.RobustEstimatePose(pts0, pts1, best_model, best_inliers, best_cost);
}

bool RobustEstimateRelativePoseHomo(const Mat3X& pts0,
                                    const Mat3X& pts1,
                                    Mat3* best_homo, 
                                    std::vector<Mat3>* rot_mat,
                                    std::vector<Vec3>* trans_vec,
                                    double inlier_threshold,
                                    double failure_probability,
                                    int max_iterations) {
  CHECK_NOTNULL(best_homo);
  CHECK_LT(failure_probability, 1.0);
  CHECK_GT(failure_probability, 0.0);
  CHECK_EQ(pts0.cols(), pts1.cols());
  
  //srand(time(NULL));

  Homography homo(4, inlier_threshold, failure_probability, max_iterations);
  homo.RobustEstimatePose(pts0, pts1, best_homo, rot_mat, trans_vec);
}

bool Homography::RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
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
  Mat3 homo, temp_homo;
  int ninliers;

  if (count == model_points_) {
    mask.fill(true);
    max_good_count = model_points_;
    if (!ComputeHomography(pts0, pts1, &homo))
      return false;
  }
  else {
    for (iter = 0; iter < niters; ++iter) {
      bool found = GetSubset2D2D(pts0, pts1, &spts0, &spts1, 1000);
      if (!found) {
        if (iter == 0)
          return false;
        break;
      }
  
      ComputeHomography(spts0, spts1, &temp_homo);
      good_count = FindInliers(pts0, pts1, temp_homo, &err, &temp_mask);
      if (good_count > std::max(max_good_count, model_points_ - 1)) {
        homo = temp_homo;
        mask = temp_mask;
        max_good_count = good_count;
        niters = UpdateRanSaCNumIters((double)(good_count)/count, niters);
      }
    }
    // Refinement
    HomoFunctorNumericalDiff functor;
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
    ComputeHomography(functor.pts0_, functor.pts1_, &homo);
    Eigen::VectorXd h(8);
    h << homo(0), homo(3), homo(6), homo(1), homo(4), homo(7), homo(2), homo(5);
    Eigen::LevenbergMarquardt<HomoFunctorNumericalDiff> lm(functor);
    lm.parameters.maxfev = pts0.cols() * 2;
    lm.parameters.xtol = 1e-14;
    lm.minimize(h);
    homo << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), 1;
  }
 
  RecoverPose(homo, pts0, pts1, &mask, &ninliers, best_model);

  if (best_cost) {
    ComputeError(pts0, pts1, homo, &err);
    *best_cost = err.sum();
  }
  if (best_inliers) {
    best_inliers->clear();
    best_inliers->resize(ninliers);
    it = best_inliers->begin();
    bool* mask_ptr = mask.data();
    for (i = 0; i < count; ++i)
      if (*(mask_ptr++) == true)
        *(it++) = i;  
  }

  return true;
}
  
bool Homography::RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
                                    Mat3* best_homo, std::vector<Mat3>* rot_mat,
                                    std::vector<Vec3>* trans_vec) {
  int count = pts0.cols(), good_count, max_good_count = 0;
  int i, iter, niters = std::max(max_iterations_, 1);
  std::vector<int>::iterator it;
  Mat3X spts0 = Mat::Ones(3, model_points_); 
  Mat3X spts1 = Mat::Ones(3, model_points_);
  Mat1X err(count); 
  Mask mask(count), temp_mask(count);
  Mat3 homo, temp_homo;
  int ninliers;

  if (count == model_points_) {
    mask.fill(true);
    max_good_count = model_points_;
    if (!ComputeHomography(pts0, pts1, &homo))
      return false;
  }
  else {
    for (iter = 0; iter < niters; ++iter) {
      bool found = GetSubset2D2D(pts0, pts1, &spts0, &spts1, 1000);
      if (!found) {
        if (iter == 0)
          return false;
        break;
      }
      ComputeHomography(spts0, spts1, &temp_homo);
      good_count = FindInliers(pts0, pts1, temp_homo, &err, &temp_mask);
      if (good_count > std::max(max_good_count, model_points_ - 1)) {
        homo = temp_homo;
        mask = temp_mask;
        max_good_count = good_count;
        niters = UpdateRanSaCNumIters((double)(good_count)/count, niters);
      }
    }
    // Refinement
    HomoFunctorNumericalDiff functor;
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
    ComputeHomography(functor.pts0_, functor.pts1_, &homo);
    Eigen::VectorXd h(8);
    h << homo(0), homo(3), homo(6), homo(1), homo(4), homo(7), homo(2), homo(5);
    Eigen::LevenbergMarquardt<HomoFunctorNumericalDiff> lm(functor);
    lm.parameters.maxfev = pts0.cols() * 2;
    lm.parameters.xtol = 1e-14;
    lm.minimize(h);
    homo << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), 1;
  }
 
  Mat3 R1, R2;
  Vec3 t1, t2;
  DecomposeHomography(homo, &R1, &R2, &t1, &t2);
  *best_homo = homo;
  if (rot_mat && trans_vec) {
    rot_mat->clear();
    rot_mat->resize(2);
    trans_vec->clear();
    trans_vec->resize(4);
    (*rot_mat)[0] = R1;
    (*rot_mat)[1] = R2;
    (*trans_vec)[0] = t1;  
    (*trans_vec)[1] = -t1;  
    (*trans_vec)[2] = t2;  
    (*trans_vec)[3] = -t2;  
  }

  return true;
}

bool Homography::ComputeHomography(const Mat3X& pts0, const Mat3X& pts1,
                                   Mat3* solution) {
  int i, j, k, count = pts0.cols();
  double icount = 1. / count;

  Eigen::Matrix<double, 9, 9> LtL, V;
  Eigen::Matrix<double, 9, 1> W;
  Vec2 cpts0(0, 0), cpts1(0, 0), spts0(0, 0), spts1(0, 0);
  
  cpts0(0) = pts0.row(0).sum() * icount;
  cpts0(1) = pts0.row(1).sum() * icount;
  cpts1(0) = pts1.row(0).sum() * icount;
  cpts1(1) = pts1.row(1).sum() * icount;
  
  spts0(0) = (pts0.row(0).array() - cpts0(0)).abs().sum();
  spts0(1) = (pts0.row(1).array() - cpts0(1)).abs().sum();
  spts1(0) = (pts1.row(0).array() - cpts1(0)).abs().sum();
  spts1(1) = (pts1.row(1).array() - cpts1(1)).abs().sum();

  if( fabs(spts0(0)) < DBL_EPSILON || fabs(spts0(1)) < DBL_EPSILON ||
      fabs(spts1(0)) < DBL_EPSILON || fabs(spts1(1)) < DBL_EPSILON )
      return false;
  
  spts0(0) = count / spts0(0);
  spts0(1) = count / spts0(1);
  spts1(0) = count / spts1(0);
  spts1(1) = count / spts1(1);

  Mat3 invHnorm, Hnorm2;
  invHnorm << 1. / spts1(0), 0, cpts1(0),
              0, 1. / spts1(1), cpts1(1),
              0, 0, 1;
  Hnorm2 << spts0(0), 0, -cpts0(0) * spts0(0),
            0, spts0(1), -cpts0(1) * spts0(1),
            0, 0, 1;

  LtL = Mat::Zero(9, 9);
  for (i = 0; i < count; ++i) {
    double x0 = (pts0(0, i) - cpts0(0)) * spts0(0);
    double y0 = (pts0(1, i) - cpts0(1)) * spts0(1);
    double x1 = (pts1(0, i) - cpts1(0)) * spts1(0);
    double y1 = (pts1(1, i) - cpts1(1)) * spts1(1);
    double Lx[] = { x0, y0, 1, 0, 0, 0, -x1 * x0, -x1 * y0, -x1 };
    double Ly[] = { 0, 0, 0, x0, y0, 1, -y1 * x0, -y1 * y0, -y1 };
    for (j = 0; j < 9; ++j)
      for (k = j; k < 9; ++k)
        LtL(j ,k) += Lx[j] * Lx[k] + Ly[j] * Ly[k];
  }
  for (i = 0; i < 9; ++i)
    for (j = 0; j < i; ++j)
      LtL(i ,j) = LtL(j, i);

  Eigen::EigenSolver<Mat> es(LtL);
  W = es.eigenvalues().real();
  V = es.eigenvectors().real();
  double min = W(0);
  for (i = 1, j = 0; i < 9; ++i) {
    double value = W(i);
    if (value < min) {
      min = value;
      j = i;
    }
  }
  double* ev = V.data() + j * 9;
  Mat3 H0;
  H0 << ev[0], ev[1], ev[2], ev[3], ev[4], ev[5], ev[6], ev[7], ev[8];
  *solution = invHnorm * H0 * Hnorm2;
  *solution *= 1. / (*solution)(2, 2);
  
  return true;
}

void Homography::RecoverPose(const Mat3& H, const Mat3X& pts0,
                             const Mat3X& pts1, Mask* mask, int* nins,
                             Mat34* Rt) {
  int i, n = pts0.cols();
  const int SOL_NUM = 4;
  std::vector<Mat34> Ps(SOL_NUM);
  std::vector<Mask> masks(SOL_NUM);

  Mat3 R1, R2;
  Vec3 t1, t2;
  DecomposeHomography(H, &R1, &R2, &t1, &t2);
  
  Ps[0] << R1, t1;
  Ps[1] << R1, -t1;
  Ps[2] << R2, t2;
  Ps[3] << R2, -t2;
  
  Mat34 P0 = Mat::Identity(3, 4);

  // Do the cheirality check.
  // Notice here a threshold dist is used to filter
  // out far away points (i.e. infinite points) since
  // there depth may vary between postive and negtive.
  Eigen::ArrayXXd dist = Mat::Constant(1, n, 50.0);
  Eigen::ArrayXXd zeros = Mat::Zero(1, n);
  Eigen::ArrayXXd Q(4, n);
  
  for (i = 0; i < SOL_NUM; ++i) {
    TriangulatePoints(P0, Ps[i], pts0, pts1, &Q);
    masks[i] = Mask(Q.row(2)> zeros);
    masks[i] = BitwiseAnd(Mask(Q.row(2) < dist), masks[i]);
    Q.block(0, 0, 3, n) = Ps[i] * Q.matrix();
    masks[i] = BitwiseAnd(Mask(Q.row(2) > zeros), masks[i]);
    masks[i] = BitwiseAnd(Mask(Q.row(2) < dist), masks[i]);
    masks[i] = BitwiseAnd(*mask, masks[i]);
  }

  int best_i = -1, best_good = -1, good;
  for (i = 0; i < SOL_NUM; ++i) {
    good = masks[i].count(); 
    if (best_good < good) {
      best_i = i;
      best_good = good;
    }
  }

  *Rt = Ps[best_i];
  *nins = best_good;
  for (i = 0; i < n; ++i)
    (*mask)(i) = masks[best_i](i);
} 

void Homography::DecomposeHomography(const Mat3& H, Mat3* R1, Mat3* R2,
                                     Vec3* t1, Vec3* t2) {
  const double epsilon = 0.001;
  Mat3 Hn, S; 
  Mat W;
  
  // Normalize homography
  Eigen::JacobiSVD<Mat> svd(H);
  W = svd.singularValues();
  Hn = H * (1.0 / W(1));

  // S = H'H - I
  S = Hn.transpose() * Hn;
  S(0, 0) -= 1.0;
  S(1, 1) -= 1.0;
  S(2, 2) -= 1.0;

  // Compute nvectors
  Vec3 np1, np2;

  double M00 = OppositeOfMinor(S, 0, 0);
  double M11 = OppositeOfMinor(S, 1, 1);
  double M22 = OppositeOfMinor(S, 2, 2);

  double rtM00 = sqrt(M00);
  double rtM11 = sqrt(M11);
  double rtM22 = sqrt(M22);

  double M01 = OppositeOfMinor(S, 0, 1);
  double M12 = OppositeOfMinor(S, 1, 2);
  double M02 = OppositeOfMinor(S, 0, 2);

  int e12 = Signd(M12);
  int e02 = Signd(M02);
  int e01 = Signd(M01);
  
  double nS00 = fabs(S(0, 0));
  double nS11 = fabs(S(1, 1));
  double nS22 = fabs(S(2, 2));

  // Find max( |Sii| ), i = 0, 1, 2
  int indx = 0;
  if (nS00 < nS11) {
    indx = 1;
    if( nS11 < nS22 )
      indx = 2;
  }
  else {
    if(nS00 < nS22 )
      indx = 2;
  }

  switch (indx) {
    case 0:
      np1(0) = S(0, 0),               np2(0) = S(0, 0);
      np1(1) = S(0, 1) + rtM22,       np2(1) = S(0, 1) - rtM22;
      np1(2) = S(0, 2) + e12 * rtM11, np2(2) = S(0, 2) - e12 * rtM11;
      break;
    case 1:
      np1(0) = S(0, 1) + rtM22,       np2(0) = S(0, 1) - rtM22;
      np1(1) = S(1, 1),               np2(1) = S(1, 1);
      np1(2) = S(1, 2) - e02 * rtM00, np2(2) = S(1, 2) + e02 * rtM00;
      break;
    case 2:
      np1(0) = S(0, 2) + e01 * rtM11, np2(0) = S(0, 2) - e01 * rtM11;
      np1(1) = S(1, 2) + rtM00,       np2(1) = S(1, 2) - rtM00;
      np1(2) = S(2, 2),               np2(2) = S(2, 2);
      break;
    default:
      break;
  }

  double traceS = S(0, 0) + S(1, 1) + S(2, 2);
  double v = 2.0 * sqrt(1 + traceS - M00 - M11 - M22);

  double ESii = Signd(S(indx, indx)) ;
  double r_2 = 2 + traceS + v;
  double nt_2 = 2 + traceS - v;

  double r = sqrt(r_2);
  double n_t = sqrt(nt_2);

  Vec3 n1 = np1 .normalized();
  Vec3 n2 = np2 .normalized();

  double half_nt = 0.5 * n_t;
  double esii_t_r = ESii * r;

  Vec3 t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
  Vec3 t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

  // R1, t1
  *R1 = Hn * (Mat::Identity(3, 3) - (2/v) * t1_star * n1.transpose());
  *t1 = *R1 * t1_star;
  if ((R1->col(0).cross(R1->col(1)) - R1->col(2)).norm() > 1)
    R1->col(2) = -R1->col(2);
  
  // R2, t2
  *R2 = Hn * (Mat::Identity(3, 3) - (2/v) * t2_star * n2.transpose());
  *t2 = *R2 * t2_star;
  if ((R2->col(0).cross(R2->col(1)) - R2->col(2)).norm() > 1)
    R2->col(2) = -R2->col(2);
}

double Homography::OppositeOfMinor(const Mat3& M, int row, int col) {
  int x1 = (col == 0)? 1: 0;
  int x2 = (col == 2)? 1: 2;
  int y1 = (row == 0)? 1: 0;
  int y2 = (row == 2)? 1: 2;

  return (M(y1, x2) * M(y2, x1) - M(y1, x1) * M(y2, x2));
}

bool Homography::CheckSubset(const Mat3X& spts0, const Mat3X& spts1) {
  const float THRESHOLD = 0.996f;
  for( int inp = 0; inp < 2; ++inp ) {
    int i, j, k = model_points_ - 1;
    const Mat3X* sptsi = inp == 0 ? &spts0 : &spts1;

    // check that the k-th selected point does not belong
    // to a line connecting some previously selected points
    for( i = 0; i < k; ++i ) {
      Vec3 d1 = (*sptsi).col(i) - (*sptsi).col(k);
      float n1 = d1(0) * d1(0) + d1(1) * d1(1);
      for(j = 0; j < i; ++j) {
        Vec3 d2 = (*sptsi).col(j) - (*sptsi).col(k);
        float denom = (d2(0) * d2(0) + d2(1) * d2(1)) * n1;
        float num = d1(0) * d2(0) + d1(1) * d2(1);
        if( num * num > THRESHOLD * THRESHOLD * denom )
          return false;
      }
    }
  }

  // We check whether the minimal set of points for the homography estimation
  // are geometrically consistent. We check if every 3 correspondences sets
  // fulfills the constraint.
  //
  // The usefullness of this constraint is explained in the paper:
  //
  // "Speeding-up homography estimation in mobile devices"
  // Journal of Real-Time Image Processing, 2013
  static const int tt[][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
  int negative = 0;
  for (int i = 0; i < 4; ++i) {
    const int* t = tt[i];
    Mat3 A, B;
    A << spts0(0, t[0]), spts0(1, t[0]), 1., 
         spts0(0, t[1]), spts0(1, t[1]), 1., 
         spts0(0, t[2]), spts0(1, t[2]), 1.;
    B << spts1(0, t[0]), spts1(1, t[0]), 1., 
         spts1(0, t[1]), spts1(1, t[1]), 1., 
         spts1(0, t[2]), spts1(1, t[2]), 1.; 
    negative += A.determinant()*B.determinant() < 0;
  }
  if (negative != 0 && negative != 4)
    return false;

  return true;
}

void Homography::ComputeError(const Mat3X& pts0, const Mat3X& pts1,
                              const Mat& homo, Mat1X* err) {
  int i, count = err->cols();
  double t = inlier_threshold_ * inlier_threshold_;
  const double* H = homo.data();
  for (i = 0; i < count; ++i) {
    double w = 1 / (H[2] * pts0(0, i) + H[5] * pts0(1, i) + H[8]);
    double dx = (H[0] * pts0(0, i) + H[3] * pts0(1, i) + H[6]) * w - pts1(0, i);
    double dy = (H[1] * pts0(0, i) + H[4] * pts0(1, i) + H[7]) * w - pts1(1, i);
    double e = dx * dx + dy * dy;
    (*err)(i) = (e > t)? t: e;
  }
}

}  // namespace rvslam
