// estimate_rot.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_ESTIMATE_ROT_H_
#define _RVSLAM_ESTIMATE_ROT_H_

#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include "rvslam_util.h"
#include "ransac.h"

namespace rvslam {
  
bool RobustEstimateRotationMat(const Mat3X& pts0,
                               const Mat3X& pts1,
                               Mat34* best_model,
                               std::vector<int>* best_inliers = NULL,
                               double *best_cost = NULL,
                               double inlier_threshold = 0.01,
                               double fail_prob = 1e-3,
                               int max_iterations = 100);

class EstimateRot : public RanSaC {

  struct RotFunctor : Functor<double> {
    int operator()(const Eigen::VectorXd &r, Eigen::VectorXd &fvec) const {
      unsigned int count = pts0_.cols();
      Mat3 R = ExpMap(r);
      for(unsigned int i = 0, j = 0; i < count; ++i) {
        double w = 1. / (R(2) * pts0_(0, i) + R(5) * pts0_(1, i) + R(8));
        fvec(j++) =   pts1_(0, i) 
                    - ((R(0) * pts0_(0, i) + R(3) * pts0_(1, i) + R(6)) * w);
        fvec(j++) =   pts1_(1, i)
                    - ((R(1) * pts0_(0, i) + R(4) * pts0_(1, i) + R(7)) * w);
      }
      return 0;
    }
  
    Mat3X pts0_;
    Mat3X pts1_;
  
    // The number of parameters of rotation matrix
    int inputs() const { return 3; }
    
    // The number of constraints
    int values() const { return pts0_.cols() * 2; }
  };

  struct RotFunctorNumericalDiff : Eigen::NumericalDiff<RotFunctor> {};

 public:
  EstimateRot(int model_points) : RanSaC(model_points) {}

  EstimateRot(int model_points, double inlier_threshold,
              double failure_probability, int max_iterations) : 
    RanSaC(model_points, inlier_threshold, failure_probability,
           max_iterations) {}

  bool RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
                          Mat34* best_model, std::vector<int>* best_inliers,
                          double *best_cost);

private:
  void ComputeRotationMat(const Mat3X&, const Mat3X&, Mat3*);
  
  void ComputeError(const Mat3X&, const Mat3X&, const Mat&, Mat1X*);

};

}  // namespace rvslam
#endif  // _RVSLAM_ESTIMATE_ROT_H_

