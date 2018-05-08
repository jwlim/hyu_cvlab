// map.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// Author: Eunate Hong(drago1301@naver.com)

#ifndef _RVSLAM_MAP_H_
#define _RVSLAM_MAP_H_

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
//#include "loopclosing.h"

namespace rvslam {
class Map { 
  private:
  
  public:

    std::map<int, Vec3> ftid_pts_map_;
    std::vector<Keyframe> keyframes_;
    bool is_keyframe_;
    bool is_success_;
    bool is_loopclosed_;
    vector<int> pose_inliers_;

    Map();
    std::map<int, Vec3> GetMap() { return ftid_pts_map_; };
    std::vector<Keyframe> GetKeyframes() { return keyframes_; };

};


}  // namespace rvslam
#endif  // _RVSLAM_MAP_H_
