// visual_odometer.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// Author: Euntae Hong(dragon1301@naver.com)

#ifndef _RVSLAM_KEYFRAME_H_
#define _RVSLAM_KEYFRAME_H_

#include <set>
#include <map>
#include <vector>
#include <list>
#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>

#include "rvslam_common.h"
#include "sim3.h"


namespace rvslam {


struct Keyframe {
  public:
  int frmno;
  Vec6 pose;
  Mat3X normalized_pts; 
  std::vector<int> ftids;
  std::vector<int> pose_inliers;  // Index to ftids and pts (sorted).
  //RelativePose2d relative_pose_2d;
  cv::Mat descriptors;
  Keyframe() : frmno(-1) {}

  //Spanning tree and Loop Edges
  bool mbFirstConnection;
  Keyframe* mpParent;
  std::set<Keyframe *> mspChildrens;
  std::set<Keyframe *> msLoopEdges;

  //Covisible keyframes
  std::map<Keyframe*, int> connectedkfs;
  std::vector<int> ordered_connectedkfs;
  std::vector<int> ordered_weights;
  int id;

  void Swap(Keyframe* keyframe) {
    std::swap(frmno, keyframe->frmno);
    pose.swap(keyframe->pose);
    normalized_pts.swap(keyframe->normalized_pts);
    ftids.swap(keyframe->ftids);
    pose_inliers.swap(keyframe->pose_inliers);
    cv::swap(descriptors,keyframe->descriptors);
    connectedkfs.swap(keyframe->connectedkfs);
    ordered_connectedkfs.swap(keyframe->ordered_connectedkfs);
    ordered_weights.swap(keyframe->ordered_weights);
    //mpParent->Swap(keyframe->mpParent);
    mspChildrens.swap(keyframe->mspChildrens);
    msLoopEdges.swap(keyframe->msLoopEdges);
    //relative_pose_2d.Swap(&keyframe->relative_pose_2d);
  }

};

inline bool Comp(int a, int b) {
  return a >b;
}

//TODO(Euntae) : Structure will be changed efficiently
//Covisible keyframes functions

inline std::set<Keyframe*> GetConnectedKeyframes(Keyframe& kf) {
  std::set<Keyframe*> s;
  for (std::map<Keyframe*, int>::iterator mit=kf.connectedkfs.begin();
       mit!=kf.connectedkfs.end();
       mit++) 
    s.insert(mit->first);
  return s;
}

//inline std::vector<Keyframe*> GetVectorCovisibleKeyframes(const Keyframe& kf) {
// return kf.ordered_connectedkfs;
//}

inline std::vector<int> GetBestCovisibilityKeyframes(const Keyframe& kf,
    const int &n) {
  if ((int)kf.ordered_connectedkfs.size() < n)
    return kf.ordered_connectedkfs;
  else
    return std::vector<int>(kf.ordered_connectedkfs.begin(),
        kf.ordered_connectedkfs.begin()+n);
}

inline std::vector<int> GetCovisiblesByWeight(Keyframe& kf,
    const int &w) {
  if (kf.ordered_connectedkfs.empty())
    return std::vector<int>();

  std::vector<int>::iterator it = upper_bound(kf.ordered_weights.begin(),
      kf.ordered_weights.end(), w,
      Comp);
  if (it == kf.ordered_weights.end())
    return std::vector<int>();
  else {
    int n = it-kf.ordered_weights.begin();
    return std::vector<int>(kf.ordered_connectedkfs.begin(),
        kf.ordered_connectedkfs.begin()+n);
  }
}


inline void AddConnection(Keyframe* cur_frm, Keyframe* kf, const int& weight) {
  if(!cur_frm->connectedkfs.count(kf))
    cur_frm->connectedkfs[kf] = weight; 
  else if(cur_frm->connectedkfs[kf] != weight)
    cur_frm->connectedkfs[kf] = weight;
  else
    return;
}
    
  
inline void UpdateCovisibleGraph(Keyframe& cur_frm){ 
  std::vector<std::pair<int, Keyframe*> > vpairs;
  vpairs.reserve(cur_frm.connectedkfs.size());
  //make vpairs
  for (std::map<Keyframe*, int>::iterator mit = 
       cur_frm.connectedkfs.begin(),
       mend=cur_frm.connectedkfs.end();
       mit!=mend; mit++) 
    vpairs.push_back(std::make_pair(mit->second, mit->first));

  sort(vpairs.begin(), vpairs.end());
  std::list<Keyframe*> lkfs;
  std::list<int> lws;
  for (int i = 0, iend=vpairs.size(); i< iend; i++) {
    lkfs.push_front(vpairs[i].second);
    lws.push_front(vpairs[i].first);
  }

  cur_frm.ordered_connectedkfs.clear();
  for (std::list<Keyframe*>::iterator it = lkfs.begin(); it != lkfs.end(); ++it) {
    cur_frm.ordered_connectedkfs.push_back((*it)->frmno);
  }
  cur_frm.ordered_weights= std::vector<int>(lws.begin(), lws.end());
}

inline void AddChild(Keyframe* cur_frm, Keyframe* kf) {
  cur_frm->mspChildrens.insert(kf);
}

inline void EraseChild(Keyframe* cur_frm, Keyframe* kf) {
  cur_frm->mspChildrens.erase(kf);
}

inline void ChangeParent(Keyframe* cur_frm, Keyframe* kf) {
  cur_frm->mpParent = kf;
  AddChild(kf, cur_frm);
}

inline int CountMatchedFeatures(const std::vector<int>& ftids1,
                                const std::vector<int>& ftids2) {
  int cnt = 0;
  for (int i = 0; i < ftids1.size(); ++i) {
    const int ftid = ftids1[i];
    for (int j = 0; j < ftids2.size(); ++j) {
      if (ftid == ftids2[j])
        cnt++;
    }
  }
  return cnt;
}

inline void SetInliersFtids(const Keyframe& kf, std::vector<int>* new_ftids) {
  new_ftids->clear();
  const std::vector<int>& inliers = kf.pose_inliers;
  const std::vector<int>& ftids = kf.ftids;
  new_ftids->reserve(inliers.size());
  for (int i = 0; i < inliers.size(); ++i) {
    const int idx = inliers[i];
    new_ftids->push_back(ftids[idx]);
  }
}


}
#endif  // _RVSLAM_KEYFRAME_H_
