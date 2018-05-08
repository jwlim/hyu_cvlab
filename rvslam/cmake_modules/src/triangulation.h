// triangulation.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_TRIANGULATION_H_
#define _RVSLAM_TRIANGULATION_H_

#include <Eigen/Dense>
#include "rvslam_common.h"

namespace rvslam {

void TriangulatePoints(const Mat34& pose0, const Mat34& pose1, 
                       const Mat3X& pts0, const Mat3X& pts1, 
                       Eigen::ArrayXXd* pts_4d);

}  // namespace rvslam

#endif  // _RVSLAM_TRIANGULATION_H_
