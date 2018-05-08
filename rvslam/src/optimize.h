// optimize.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// Author: Eunate Hong(drago1301@naver.com)

#ifndef _RVSLAM_OPTIMIZE_H_
#define _RVSLAM_OPTIMIZE_H_

#include <algorithm>
#include <map>
#include <vector>
#include <glog/logging.h>
#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>

#include "rvslam_common.h"
#include "vehicle_info.h"
#include "keyframe.h"
#include "voctree.h"
#include "rvmap.h"
#include "visual_odometer.h"
#include "loopclosing.h"

//#include "loopclosing.h"

namespace rvslam {
class Optimize { 
  private:
    Map *world_;
    bool *is_keyframe_;
    void LocalOptimizeWorld();
    float pose_inlier_thr_;
    void FindPoseInliers(const vector<int>& ftids,
                         const Mat3X& normalized_pts,
                         const Mat3X& world_pts,
                         const Vec6& pose,
                         const double inlier_thr,
                         vector<int>* inliers) const; 
    void UpdateWorldPoints();
    void FindMatchedPoints(const Mat3X& pts1, const vector<int>& ftids1,
                           const Mat3X& pts2, const vector<int>& ftids2,
                           Mat3X* new_pts1, Mat3X* new_pts2,
                           vector<int>* new_ftids) const ;
    LoopClosing loop;
  
  public:
    Optimize(const char *voc_file, VisualOdometer::Calib calib,
             Map* world);
    bool Process();
    set<int> lba_ftids_;

};


}  // namespace rvslam
#endif  // _RVSLAM_OPTIMIZE_H_
