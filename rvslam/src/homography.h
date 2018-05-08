// homography.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_HOMOGRAPHY_H_
#define _RVSLAM_HOMOGRAPHY_H_

#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include "rvslam_common.h"
#include "ransac.h"

namespace rvslam {

  
bool RobustEstimateRelativePoseHomo(const Mat3X& pts0,
                                    const Mat3X& pts1,
                                    Mat34* best_model,
                                    std::vector<int>* best_inliers = NULL,
                                    double *best_cost = NULL,
                                    double inlier_threshold = 0.01,
                                    double fail_prob = 1e-3,
                                    int max_iterations = 100);

bool RobustEstimateRelativePoseHomo(const Mat3X& pts0,
                                    const Mat3X& pts1,
                                    Mat3* best_homo, 
                                    std::vector<Mat3>* rot_mat = NULL,
                                    std::vector<Vec3>* trans_vec = NULL,
                                    double inlier_threshold = 0.01,
                                    double fail_prob = 1e-3,
                                    int max_iterations = 100);

class Homography : public RanSaC {

  struct HomoFunctor : Functor<double> {
    int operator()(const Eigen::VectorXd &h, Eigen::VectorXd &fvec) const {
      unsigned int count = pts0_.cols();
      for(unsigned int i = 0, j = 0; i < count; ++i) {
        double w = 1. / (h(6) * pts0_(0, i) + h(7) * pts0_(1, i) + 1);
        fvec(j++) =   pts1_(0, i) 
                    - ((h(0) * pts0_(0, i) + h(1) * pts0_(1, i) + h(2)) * w);
        fvec(j++) =   pts1_(1, i)
                    - ((h(3) * pts0_(0, i) + h(4) * pts0_(1, i) + h(5)) * w);
      }
      return 0;
    }
  
    Mat3X pts0_;
    Mat3X pts1_;
  
    // The number of parameters of homography
    int inputs() const { return 8; }
    
    // The number of constraints
    int values() const { return pts0_.cols() * 2; }
  };

  struct HomoFunctorNumericalDiff : Eigen::NumericalDiff<HomoFunctor> {};

 public:
  Homography(int model_points) : RanSaC(model_points) {}

  Homography(int model_points, double inlier_threshold,
             double failure_probability, int max_iterations) : 
    RanSaC(model_points, inlier_threshold, failure_probability,
           max_iterations) {}

  bool RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
                          Mat34* best_model, std::vector<int>* best_inliers,
                          double *best_cost);
  
  bool RobustEstimatePose(const Mat3X& pts0, const Mat3X& pts1,
                          Mat3* best_homo, std::vector<Mat3>* rot_mat,
                          std::vector<Vec3>* trans_vec);

private:
  bool ComputeHomography(const Mat3X&, const Mat3X&, Mat3*);
 
  void RecoverPose(const Mat3&, const Mat3X&, const Mat3X&, Mask*, int*,
                   Mat34*);

  void DecomposeHomography(const Mat3&, Mat3*, Mat3*, Vec3*, Vec3*);
  
  double OppositeOfMinor(const Mat3&, int, int);
  
  bool CheckSubset(const Mat3X&, const Mat3X&);
  
  void ComputeError(const Mat3X&, const Mat3X&, const Mat&, Mat1X*);

  int Signd(const double x) { return ((x >= 0)? 1: -1); }

};

}  // namespace rvslam
#endif  // _RVSLAM_HOMOGRAPHY_H_
