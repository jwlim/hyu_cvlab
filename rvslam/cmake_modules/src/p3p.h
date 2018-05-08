// p3p.h
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_P3P_H_
#define _RVSLAM_P3P_H_

#include <vector>
#include "rvslam_common.h"

namespace rvslam {

int ComputePosesP3P(const Mat3X& image_points, const Mat3X& world_points,
                    std::vector<Mat34>* solutions);

bool RobustEstimatePoseP3P(const Mat3X& image_points,
                           const Mat3X& world_points,
                           double inlier_threshold,
                           Mat34* best_model_ret,
                           std::vector<int> *best_inliers = NULL,
                           double *best_score = NULL,
                           double failure_probability = 1e-3,
                           size_t max_iterations = 100);

}  // namespace rvslam
#endif  // _RVSLAM_P3P_H_
