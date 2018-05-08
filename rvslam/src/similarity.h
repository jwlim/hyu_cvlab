// similarity.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_SIMILARITY_H_
#define _RVSLAM_SIMILARITY_H_

#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include "rvslam_util.h"
#include "ransac.h"

namespace rvslam {

  
bool RobustEstimateSimilarity(const Mat3X& pts2d0,
                              const Mat3X& pts2d1,
                              const Mat3X& pts3d0,
                              const Mat3X& pts3d1,
                              Mat34* best_model,
                              std::vector<int>* best_inliers = NULL,
                              double *best_cost = NULL,
                              double *scale = 0,
                              double inlier_threshold = 0.01,
                              double fail_prob = 1e-3,
                              int max_iterations = 100);

class Similarity : public RanSaC {
 public:
  Similarity(int model_points) : RanSaC(model_points) {}

  Similarity(int model_points, double inlier_threshold,
             double failure_probability, int max_iterations) : 
    RanSaC(model_points, inlier_threshold, failure_probability,
           max_iterations) {}

  bool RobustEstimatePose(const Mat3X& pts2d0, const Mat3X& pts2d1,
                          const Mat3X& pts3d0, const Mat3X& pts3d1,
                          Mat34* best_model, std::vector<int>* best_inliers,
                          double *best_cost);
  inline double GetScale() { return scale_; };

private:
  void ComputeSimilarity(const Mat3X&, const Mat3X&, Mat34*);
  
  Mat34 InverseSimilarity(const Mat34&);

  void ComputeError(const Mat3X&, const Mat3X&, const Mat&, Mat1X*);
  double scale_;

};

}  // namespace rvslam
#endif  // _RVSLAM_SIMILARITY_H_

