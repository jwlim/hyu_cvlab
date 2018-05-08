// map.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
//         Euntae Hong(dragon1301@naver.com)

#include <map>
#include <set>
#include <list>

#include "rvmap.h"

#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;


const int kNumKeyframes = 5000;

using namespace std;

namespace rvslam {
  Map::Map()
    : keyframes_(), ftid_pts_map_() {
      keyframes_.reserve(kNumKeyframes + 1);
      keyframes_.resize(1);

    }




}

