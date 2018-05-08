#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "time.h"

#include "rvslam_util.h"
#include "rvslam_profile.h"

#include "csio/csio_stream.h"
#include "csio/csio_frame_parser.h"

namespace rvslam {

class Csio {
  private:
    csio::OutputStream csio_out;
    std::ofstream ofs_pose_out;
    Vec6 prev_pose;

  public:
    void Process(const int idx, const Mat3X& image_pts, 
                 const Vec6& cur_pose, const std::vector<Vec6>& pose_vec,
                 const std::vector<int> indices, const Mat3X& pts_3d);

};


};
