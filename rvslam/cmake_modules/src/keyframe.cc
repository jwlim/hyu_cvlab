// keyframe.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
//         Euntae Hong(dragon1301@naver.com)

#include "visual_odometer.h"

#include <stdio.h>
#include <algorithm>
#include <iomanip>  // setfill(' ')
#include <map>
#include <set>
#include <list>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "p3p.h"
#include "rvslam_common.h"
#include "rvslam_util.h"
#include "five_point.h"
#include "triangulation.h"
#include "homography.h"
#include "estimate_rot.h"

#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;

//#define USE_QUATERNION

using namespace std;

