// five_point.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_FIVE_POINT_H_
#define _RVSLAM_FIVE_POINT_H_

#include <vector>
#include <Eigen/Dense>
#include "rvslam_common.h"
#include "ransac.h"

namespace rvslam {

bool RobustEstimateRelativePose5pt(const Mat3X& pts0,
                                   const Mat3X& pts1,
                                   Mat34* best_model,
                                   std::vector<int>* best_inliers = NULL,
                                   double *best_cost = NULL,
                                   double inlier_threshold = 0.01,
                                   double fail_prob = 1e-3,
                                   int max_iterations = 100);

bool RobustEstimateRelativePose5pt(const Mat3X& pts0,
                                   const Mat3X& pts1,
                                   Mat3* best_em, 
                                   std::vector<Mat3>* rot_mat = NULL,
                                   std::vector<Vec3>* trans_vec = NULL,
                                   double inlier_threshold = 0.01,
                                   double fail_prob = 1e-3,
                                   int max_iterations = 100);

class FivePoint : public RanSaC {
 
  typedef std::complex<double> Complex;
  
 public:
  FivePoint(int model_points) : RanSaC(model_points) {}

  FivePoint(int model_points, double inlier_threshold,
            double failure_probability, int max_iterations) : 
    RanSaC(model_points, inlier_threshold, failure_probability,
           max_iterations) {}

  bool RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
                          Mat34* best_model,
                          std::vector<int>* best_inliers = NULL, 
                          double *best_cost = NULL);
  
  bool RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
                          Mat3* best_em, std::vector<Mat3>* rot_mat,
                          std::vector<Vec3>* trans_vec);
  
 private:
  void ComputeEssentialMat(const Mat3X&, const Mat3X&, std::vector<Mat3>*);
  
  void GetCoeffMat(double*, double*);

  void RecoverPose(const Mat3&, const Mat3X&, const Mat3X&, Mask*, int*,
                   Mat34*);
  
  void DecomposeEssentialMat(const Mat3&, Mat3*, Mat3*, Vec3*);
  
  void ComputeError(const Mat3X&, const Mat3X&, const Mat&, Mat1X*);

};

}  // namespace rvslam
#endif  // _RVSLAM_FIVE_POINT_H_
