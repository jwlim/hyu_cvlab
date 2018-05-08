// loopcloing.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// Author: Euntae Hong (dragon1301@naver.com)
//

#ifndef _RVSLAM_LOOPCLOSING_H_
#define _RVSLAM_LOOPCLOSING_H_

#include <fstream>
#include <map>
#include <vector>

#include "keyframe.h"
#include "voctree.h"
#include "sim3.h"

#include <glog/logging.h>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

namespace rvslam {

typedef map<Keyframe*, Sim3, std::less<Keyframe*>,
        Eigen::aligned_allocator<std::pair<const Keyframe*, Sim3> > > KeyFrameAndPose;

class LoopClosing {
  private:
    voctree_t<40, 3, 32> voc;
    int id_;
    cv::Mat calib_;
    cv::Mat rel_pose_;
    double inlier_threshold_;
    map<int, Keyframe*> kf_id;
    vector<int> loopcandidates_;
    Keyframe* matchedkf_;
    Sim3 scw_;
    int num_edge_;
    std::map<int, int> covisible_id;

    bool DetectLoop(Keyframe& cur_kf, vector<Keyframe>& keyframes);
    bool ComputeSim3(Keyframe& cur_kf, vector<Keyframe>& keyframes,
                     const std::map<int, Vec3>& ftid_pts_map); 
    bool CorrectLoop(Keyframe& cur_kf,
                     vector<Keyframe>& keyframes,
                     std::map<int, Vec3>& ftid_pts_map);
    void PoseGraphOptimization(vector<Keyframe>& keyframes,
                               map<int,Vec3>& ftid_pts_map,
                               map<Keyframe*, vector<int> > loopconnections,
                               KeyFrameAndPose &noncorrectedsim3,
                               KeyFrameAndPose &correctedsim3);
    bool UpdateConnections(vector<Keyframe>& keyframes, int num_keyframes,
                           Keyframe* cur_frm);

  public:
    bool Process(Keyframe& cur_kf,
                 vector<Keyframe>& keyframes,
                 map<int, Vec3>& ftid_pts_map);
    void Setup(const char *voc_file, float fx, float fy, float cx, float cy,
               double inlier_threshold);
};







}

#endif  // _RVSLAM_LOOPCLOSING_H_
