// loopclosing.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// Author: Euntae Hong(dragon1301@naver.com)
//

#include <math.h>
#include <stdarg.h>

#include <map>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "p3p.h"
#include "keyframe.h"
#include "loopclosing.h"
#include "similarity.h"
#include "sim3solver.h"
#include "feat_opencv.h"
#include "sim3.h"
#include "visual_odometer.h"

#include "rvslam_util.h"
#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;

DEFINE_int32(loop_th, 50, "Number of minimum mathcing.");

using namespace std;


namespace rvslam {

template <typename T>
class TSim3 {
  typedef Eigen::Quaternion<T> TQuat;
  typedef Eigen::Matrix<T, 3, 1> TVec3;
  typedef Eigen::Matrix<T, 3, 3> TMat3;
    
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  TSim3() {
    r_ = TQuat(T(1), T(0), T(0), T(0));
    t_.fill(T(0));
    s_ = T(1);
  }

  TSim3(const rvslam::Sim3& S) {
    Eigen::Quaterniond r = S.Rotation();
    rvslam::Vec3 t = S.Translation();
    r_ = TQuat(T(r.w()), T(r.x()), T(r.y()), T(r.z()));
    t_ << T(t[0]), T(t[1]), T(t[2]);
    s_ = T(S.Scale());
  }

  TSim3(const TQuat& r, const TVec3& t, T s)
      : r_(r), t_(t), s_(s) {
    r_.normalize();
  }

  TSim3(const T* spose) {
    const T& a0 = spose[0];
    const T& a1 = spose[1];
    const T& a2 = spose[2];
    const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    T q0, q1, q2, q3;
    if (theta_squared > T(1e-5)) {
      const T theta = sqrt(theta_squared);
      const T half_theta = theta * T(0.5);
      const T k = sin(half_theta) / theta;
      q0 = cos(half_theta);
      q1 = a0 * k;
      q2 = a1 * k;
      q3 = a2 * k;
    } else {
      const T k(0.5);
      q0 = T(1.0);
      q1 = a0 * k;
      q2 = a1 * k;
      q3 = a2 * k;
    }
    r_ = TQuat(q0, q1, q2, q3);
    t_[0] = spose[3]; 
    t_[1] = spose[4]; 
    t_[2] = spose[5]; 
    s_ = spose[6]; 
  }

  void GetLog(T* u) const {
    T eps(1e-5);
    T lambda = log(s_);
    T lambda2 = lambda * lambda;
    T lambda_abs = (lambda < T(0))? -lambda: lambda;

    T w[3], q[4];
    q[0] = r_.w(); 
    q[1] = r_.x(); 
    q[2] = r_.y(); 
    q[3] = r_.z(); 
    ceres::QuaternionToAngleAxis(q, w);
    const T& w0 = w[0];
    const T& w1 = w[1];
    const T& w2 = w[2];
    TMat3 W;
    W << T(0), -w2, w1, w2, T(0), -w0, -w1, w0, T(0);
    TMat3 W2 = W * W;
    T a2 = w0 * w0 + w1 * w1 + w2 * w2; 
    // The derivative of sqrt(0) (i.e., sqrt'(0)) is infinite.
    // Be sure to avoid it.
    T a = (a2 < eps * eps)? T(0): sqrt(a2); 
    T sa = sin(a);
    T ca = cos(a);
    TMat3 I = TMat3::Identity();
    TMat3 R = r_.toRotationMatrix();
    
    T c0, c1, c2;
    if (a < eps) {
      R = I + W + T(0.5) * W2;
      if (lambda_abs < eps)
        c0 = s_;
      else
        c0 = (s_ - T(1)) / lambda;
      c1 = (T(3) * s_ * ca - lambda * s_ * ca - a * s_ * sa) / T(6);
      c2 = s_ * a / T(6) - lambda * s_ * ca / T(24);
    }
    else {
      R = I +  W * sa / a + W2 * (T(1) - ca) / a2;
      if (lambda_abs < eps) {
        c0 =   s_;
        c1 =   (T(2) * s_ * sa - a * s_ * ca + lambda * s_ * sa) / (T(2) * a);
        c2 =   s_ / a2 - s_ * sa / (T(2) * a) 
             - (T(2) * s_ * ca + lambda * s_ * ca) / (T(2) * a2);
      }
      else {
        c0 =   (s_ - T(1)) / lambda;
        c1 =   (a * (T(1) - s_ * ca) + s_ * lambda * sa) / (a * (lambda2 + a2));
             - (lambda * (s_ * ca - T(1))) / (a2 * (lambda2 + a2));
      }
    }
    TMat3 V = c0 * I + c1 * W + c2 * W2;
    TVec3 v = V.lu().solve(t_);

    for (int i = 0; i < 3; ++i) {
      u[i] = v[i];
      u[i + 3] = w[i];
    }
    u[6] = lambda;
  }

  TSim3 Inverse() const {
    return TSim3(r_.conjugate(), r_.conjugate() * ((-1. / s_) * t_), 1. / s_);
  }

  TSim3 operator*(const TSim3& u) const {
    TSim3 result;
    result.r_ = r_ * u.r_;
    result.t_ = s_ * (r_ * u.t_) +t_;
    result.s_ = s_ * u.s_;
    return result;
  }

 protected:
  TQuat r_;
  TVec3 t_;
  T s_;
};

struct SPoseErrorFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SPoseErrorFunctor(rvslam::Sim3 S) : S(S) {} 

  template <typename T>
  bool operator()(const T* const spose0,
                  const T* const spose1,
                  T* residuals) const {
    // Because S10 = S1 * S0^-1
    // --> S10 * S0 * S1^-1 should be Identity
    TSim3<T> S10(S);
    TSim3<T> S0(spose0);
    TSim3<T> S1(spose1);
    TSim3<T> Serr = S10 * S0 * S1.Inverse();
    Serr.GetLog(residuals);
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const rvslam::Sim3 S) {
    return (new ceres::AutoDiffCostFunction<SPoseErrorFunctor, 7, 7, 7>(
                new SPoseErrorFunctor(S)));
  }

  rvslam::Sim3 S;
}
;
//Define Function
Vec6 GetPoseFromSim3(const double* spose);

void SetKeyframeID(vector<Keyframe>& keyframes);

void SetKeypoints(const Mat3X& pts, vector<cv::KeyPoint>* keypoints);

int FindKeyframe(const int no, vector<Keyframe>& keyframes);

void MergeMap(const Mat3X& normalized_pts,
              const map<int, Vec3>& ftid_pts_map,
              map<int, int>& ftid_matching,
              vector<int>* new_ftids,
              Mat3X* new_normalized_pts,
              vector<int>* ftid_idx_found); 

void BuildWorldPointsMatrixForP3P(const vector<int>& ftids,
                                  const map<int, Vec3>& ftid_pts_map,
                                  vector<int>* ftid_idx_found,
                                  Mat3X* world_pts);

inline std::string StringPrintf(const char* fmt, ...) {
  char buf[1024];
  va_list a;
  va_start(a, fmt);
  vsprintf(buf, fmt, a);
  va_end(a);
  return std::string(buf);
}

bool LoopClosing::Process(Keyframe& cur_kf,
                          vector<Keyframe>& keyframes,
                          map<int, Vec3>& ftid_pts_map) {
  int id;
  if (DetectLoop(cur_kf, keyframes)) {
    if (ComputeSim3(cur_kf, keyframes, ftid_pts_map)) {
      if (CorrectLoop(cur_kf, keyframes, ftid_pts_map)) {
        return true;
      }
    }
  }

  vector<float*> featdesc;
  cv::Mat& descriptors = cur_kf.descriptors;
  for (int i = 0; i < descriptors.rows; ++i) {
		featdesc.push_back(descriptors.ptr<float>(i,0));
  }
  vector<pair<float, int> > doc_score;
  voc.insert_doc(cur_kf.frmno, featdesc);

  kf_id.insert(make_pair(cur_kf.frmno, &cur_kf));
  return false;
}

void LoopClosing::Setup(const char *voc_file, float fx, float fy,
                        float cx, float cy, double inlier_threshold) {
  voc.load(voc_file);
  calib_ = (cv::Mat_<float>(2,2) << fx, fy, cx, cy);
  inlier_threshold_ = inlier_threshold;
}

bool LoopClosing::ComputeSim3(Keyframe& cur_kf,
                              vector<Keyframe>& keyframes,
                              const map<int, Vec3>& ftid_pts_map) {
  Mat3X best_pts2d1, best_pts2d2;
  Mat3X best_pts3d1, best_pts3d2;
  int matching_th = FLAGS_loop_th;
  //TODO : added to test
  Mat34 best_pose_mat;


  //Select best kf on loopcandidates
  const Mat3X& img_pts1 = cur_kf.normalized_pts;
  cv::Mat& des1 = cur_kf.descriptors;
  vector<cv::KeyPoint> kps1;
  SetKeypoints(img_pts1, &kps1);
  vector<int>& ftids1 = cur_kf.ftids;
  vector<int> best_p3p_inliers;
  vector<int> best_merge_idx;
  vector<int> best_ftids;
  Mat34 best_p3p_pose;
  for (int i = 0; i < loopcandidates_.size(); ++i) {
    map<int, int> ftid_matching;
    Keyframe ref_kf = keyframes[loopcandidates_[i]];
    cv::Mat& des2 = ref_kf.descriptors;
    const Mat3X& img_pts2 = ref_kf.normalized_pts;
    const vector<int>& ftids2 = ref_kf.ftids;
    vector<cv::KeyPoint> kps2;
    SetKeypoints(img_pts2, &kps2);

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches;
    float rad = 50 / 718;

    //Matching
    RadiusMatching(des1, kps1, des2, kps2, rad, 30, &matches);
    //matcher.match(des1, des2, matches);

    const int n = matches.size();
    Mat3X pts2d1(3,n), pts2d2(3,n);
    Mat3X pts3d1(3,n), pts3d2(3,n);

    if (matches.size() < 30) { 
      continue;
    }

    for (int j = 0; j < matches.size(); j++) {
      //q = ref, t = cur 
      const int q = matches[j].queryIdx;
      const int t = matches[j].trainIdx;
      const int ftid1 = ftids1[t];
      const int ftid2 = ftids2[q];
      map<int, Vec3>::const_iterator it = ftid_pts_map.find(ftid1);
      map<int, Vec3>::const_iterator it2 = ftid_pts_map.find(ftid2);
      if (it == ftid_pts_map.end() || it2 == ftid_pts_map.end())
        continue;
      Vec3 p1 = it->second;
      Vec3 p2 = it2->second;
      ftid_matching.insert(make_pair(t, ftid2));

      pts2d1.col(j) = img_pts1.col(t);
      pts2d2.col(j) = img_pts2.col(q);
      pts3d1.col(j) = p1;
      pts3d2.col(j) = p2;
    }

    vector<int> new_ftids;
    Mat3X new_normalized_pts;
    vector<int> merge_idx_found;
    MergeMap(img_pts1, ftid_pts_map, ftid_matching,
             &new_ftids, &new_normalized_pts, &merge_idx_found);

    // Perform P3P RANSAC.
    vector<int> ftid_idx_found;
    Mat3X world_pts; 
    BuildWorldPointsMatrixForP3P(new_ftids, ftid_pts_map, &ftid_idx_found, &world_pts);
    Mat34 pose_mat;
    vector<int> pose_inliers;
    RobustEstimatePoseP3P(new_normalized_pts, world_pts,
                          inlier_threshold_, &pose_mat, &pose_inliers,
                          NULL,1e-4); 

    if (pose_inliers.size() > best_p3p_inliers.size()) {
      //TODO : added to test
      best_pose_mat = pose_mat;

      best_p3p_inliers = pose_inliers;
      id_ = loopcandidates_[i];
      best_pts2d1 = pts2d1;
      best_pts2d2 = pts2d2;
      best_pts3d1 = pts3d1;
      best_pts3d2 = pts3d2;
      best_p3p_pose = pose_mat;
      best_merge_idx = merge_idx_found;
      best_ftids = new_ftids;
    }
  }

  //p3p validation
  cout << keyframes[id_].frmno <<"th frame, " << "p3p inliers : " << best_p3p_inliers.size() << endl;
  if (best_p3p_inliers.size() <= matching_th)
    return false;

  Mat34 best_model;
  vector<int> best_inliers;
  double best_cost;
  double inlier_threshold;
  double best_scale;
  double threshold = 3;
  RobustEstimateSimilarity(best_pts2d1, best_pts2d2,
                           best_pts3d1, best_pts3d2, &best_model,
                           &best_inliers, &best_cost, &best_scale,
                           1);


  cout << "sim3 inliers : " << best_inliers.size() << endl;
  //if (best_inliers.size() <= matching_th)
  //  return false;

  //change ftids based on loop matched keyframe
  for (int i = 0; i < best_merge_idx.size(); ++i) {
    const int idx = best_merge_idx[i];
    ftids1[idx] = best_ftids[i]; 
  }

  Mat3 r = best_model.block(0,0,3,3);
  Vec3 t = best_model.block(0,3,3,1);
  const double s = 1;
  Sim3 gscm(r,t,s);
  matchedkf_ = &keyframes[id_];
  const Mat3 r_cur = best_p3p_pose.block(0,0,3,3);
  const Vec3 t_cur = best_p3p_pose.block(0,3,3,1);
  const double s_cur = best_scale;

  Sim3 gsmw(r_cur, t_cur, s_cur);
  scw_ = gsmw;
  
  return true;
}

void MergeMap(const Mat3X& normalized_pts,
              const map<int, Vec3>& ftid_pts_map,
              map<int, int>& ftid_matching,
              vector<int>* new_ftids,
              Mat3X* new_normalized_pts,
              vector<int>* ftid_idx_found) {
  new_ftids->clear();
  ftid_idx_found->clear();
  new_ftids->reserve(ftid_matching.size());
  new_normalized_pts->resize(3, ftid_matching.size());
  int idx = 0;
  for (map<int,int>::iterator it = ftid_matching.begin();
       it != ftid_matching.end(); it++) {
    int ftid = it->second;
    new_ftids->push_back(ftid);
    new_normalized_pts->col(idx) = normalized_pts.col(it->first);
    ftid_idx_found->push_back(it->first);
    idx++;
  }
}


void SetKeypoints(const Mat3X& pts, vector<cv::KeyPoint>* keypoints) {
  const int n = pts.cols();
  keypoints->reserve(n);

  for (int i = 0; i < n; ++i) {
    cv::KeyPoint kp;
    kp.pt.x = pts(0,i);
    kp.pt.y = pts(1,i);
    kp.size = 1;
    kp.angle = 0;
    keypoints->push_back(kp);
  }
}



bool LoopClosing::DetectLoop(Keyframe& cur_kf, vector<Keyframe>& keyframes) {
  if (keyframes.size() < 10)
    return false;

  //Check similarity
  vector<float*> featdesc;
  cv::Mat& descriptors = cur_kf.descriptors;
  vector<pair<float, int> > doc_score;
  for (int i = 0; i < descriptors.rows; ++i) {
    featdesc.push_back(descriptors.ptr<float>(i,0));
  }
  voc.query_doc(featdesc, doc_score);
  const int frmno = doc_score[0].second;
  const int idx = FindKeyframe(frmno, keyframes);
  const int diff = abs(cur_kf.frmno - frmno);
  bool loop_detected = false;
  const int diff_th = 300;

  if (diff > diff_th){
    loop_detected = true;
  }
  else 
    return false;
  
  id_ = idx;
  loopcandidates_.clear();
  loopcandidates_.reserve(doc_score.size());
  int num_candidates = 10;
  if (num_candidates > doc_score.size())
    num_candidates = doc_score.size();
  for (int i = 0; i < num_candidates; ++i) {
    const int frmno = doc_score[i].second;
    const int diff = abs(cur_kf.frmno - frmno);
    if (diff < diff_th)
      continue;
    const int idx = FindKeyframe(frmno, keyframes);
    loopcandidates_.push_back(idx);
  }

  return true;
}

int FindKeyframe(const int no, vector<Keyframe>& keyframes) {
  for (int i = 0; i < keyframes.size(); ++i) {
    if (keyframes[i].frmno == no)
      return i;
  }
}

void BuildWorldPointsMatrixForP3P(const vector<int>& ftids,
                                  const map<int, Vec3>& ftid_pts_map,
                                  vector<int>* ftid_idx_found,
                                  Mat3X* world_pts) {
  CHECK_NOTNULL(ftid_idx_found);
  CHECK_NOTNULL(world_pts);
  // Initialize ftid_idx_found, world_pts, and normalized_image_pts.
  ftid_idx_found->clear();
  if (ftid_pts_map.empty()) return;
  const int num_ftids = ftids.size();
  ftid_idx_found->reserve(num_ftids);
  world_pts->resize(3, num_ftids);
  // Find image_pts that has matched world_pts in ftid_pts_map_.
  for (int i = 0; i < ftids.size(); ++i) {
    map<int, Vec3>::const_iterator it = ftid_pts_map.find(ftids[i]);
    if (it == ftid_pts_map.end()) continue;
    const int col_idx = ftid_idx_found->size();
    ftid_idx_found->push_back(i);
    world_pts->col(col_idx) = it->second;
  }
  world_pts->conservativeResize(3, ftid_idx_found->size());
}


bool LoopClosing::CorrectLoop(Keyframe& cur_kf,
                              vector<Keyframe>& keyframes,
                              map<int, Vec3>& ftid_pts_map) {

  int matching_th = FLAGS_loop_th;
  //Update Covisible graph
  for (int i = 0; i < keyframes.size(); ++i) {
    Keyframe& kf = keyframes[i];
    if(!UpdateConnections(keyframes, keyframes.size(), &kf)) {
      cout << "Connection fails" << endl;
      return false;
    }
  }


  //Merge Landmarks on covisible graph of cur_kf
  const Mat3X& img_pts1 = cur_kf.normalized_pts;
  cv::Mat& des1 = cur_kf.descriptors;
  vector<cv::KeyPoint> kps1;
  SetKeypoints(img_pts1, &kps1);
  const vector<int>& ftids1 = cur_kf.ftids;
  int best_p3p_inliers = matching_th;
  vector<int>& covisible_graph = cur_kf.ordered_connectedkfs;

  for (int i = 0; i < covisible_graph.size(); ++i) {
    map<int, int> ftid_matching;
    const int frmno = covisible_graph[i];
    const int kfidx = FindKeyframe(frmno, keyframes);
    Keyframe& ref_kf = keyframes[kfidx];
    cv::Mat& des2 = ref_kf.descriptors;
    const Mat3X& img_pts2 = ref_kf.normalized_pts;
    vector<int>& ftids2 = ref_kf.ftids;
    vector<cv::KeyPoint> kps2;
    SetKeypoints(img_pts2, &kps2);

    vector<cv::DMatch> matches;
    float rad = 100 / 718;

    //Matching
    RadiusMatching(des1, kps1, des2, kps2, rad, 30, &matches);

    const int n = matches.size();
    Mat3X pts2d1(3,n), pts2d2(3,n);
    Mat3X pts3d1(3,n), pts3d2(3,n);

    if (matches.size() < 30) { 
      continue;
    }

    for (int j = 0; j < matches.size(); j++) {
      //q = ref, t = cur 
      const int q = matches[j].queryIdx;
      const int t = matches[j].trainIdx;
      const int ftid1 = ftids1[t];
      const int ftid2 = ftids2[q];
      map<int, Vec3>::const_iterator it = ftid_pts_map.find(ftid1);
      map<int, Vec3>::const_iterator it2 = ftid_pts_map.find(ftid2);
      if (it == ftid_pts_map.end() || it2 == ftid_pts_map.end())
        continue;
      Vec3 p1 = it->second;
      Vec3 p2 = it2->second;
      ftid_matching.insert(make_pair(q, ftid1));

      pts2d1.col(j) = img_pts1.col(t);
      pts2d2.col(j) = img_pts2.col(q);
      pts3d1.col(j) = p1;
      pts3d2.col(j) = p2;
    }

    vector<int> new_ftids;
    Mat3X new_normalized_pts;
    vector<int> merge_idx_found;
    MergeMap(img_pts2, ftid_pts_map, ftid_matching,
             &new_ftids, &new_normalized_pts, &merge_idx_found);

    // Perform P3P RANSAC.
    Mat3X world_pts; 
    vector<int> ftid_idx_found;
    BuildWorldPointsMatrixForP3P(new_ftids, ftid_pts_map, &ftid_idx_found, &world_pts);
    Mat34 pose_mat;
    vector<int> pose_inliers;
    RobustEstimatePoseP3P(new_normalized_pts, world_pts,
                          inlier_threshold_, &pose_mat, &pose_inliers,
                          NULL, 1e-4); 

    if (pose_inliers.size() < 30)
      continue;

    for (int j = 0; j < merge_idx_found.size(); ++j) {
      const int idx = merge_idx_found[j];
      ftids2[idx] = new_ftids[j];
    }
  } 


  KeyFrameAndPose correctedsim3, noncorrectedsim3;
  correctedsim3[&cur_kf] = scw_; 
  Mat34 cur_pose = VisualOdometer::ToPoseMatrix(cur_kf.pose);

  Mat3 riw = cur_pose.block(0,0,3,3);
  Vec3 tiw_vec = cur_pose.block(0,3,3,1);
  Sim3 siw(riw, tiw_vec, 1.0);

  // pose without correction
  noncorrectedsim3[&cur_kf] = siw;

  file.open(filename2);
  file << keyframes.size() << endl;
  for (int i = keyframes.size()-1; i >= 0 ; --i){
    const Keyframe &kf2 = keyframes[i];
    const vector<int>& connectedkfs = kf2.ordered_connectedkfs;
    file << connectedkfs.size() << endl;
    for (int j = 0; j < connectedkfs.size(); ++j) {
      const int frmno = connectedkfs[j];
      const int idx = FindKeyframe(frmno, keyframes);
      file << i << " " << idx << endl;
    }
  }
  file.close();

  file.open(filename3);
  for (int i = keyframes.size()-1; i >= 0 ; --i){
    file << keyframes[i].frmno << endl;
  }
  file.close();
  }

  vector<int> ordered_connectedkfs = cur_kf.ordered_connectedkfs;
  //new links and make loopconnections
  map<Keyframe*, vector<int> > loopconnections;
  const int total = keyframes.size();
  for (int i = 0; i < ordered_connectedkfs.size(); ++i) {
    const int idx = ordered_connectedkfs[i];
    const int kfidx = FindKeyframe(idx, keyframes);
    Keyframe& kf = keyframes[kfidx];
    loopconnections[&kf] = kf.ordered_connectedkfs;
  }

  SetKeyframeID(keyframes);
  PoseGraphOptimization(keyframes,
                        ftid_pts_map,
                        loopconnections,
                        noncorrectedsim3,
                        correctedsim3);
  return true;
}


bool LoopClosing::UpdateConnections(vector<Keyframe>& keyframes, int num_keyframes,
                                    Keyframe* cur_frm) {
  map<Keyframe*, int> kfcounter;
  if (num_keyframes > keyframes.size())
    num_keyframes = keyframes.size();
  vector<int> cur_ftids;
  SetInliersFtids(*cur_frm, &cur_ftids);
  
  //Check covisible keyframes which are sharing same feature ID
  for (int i = 0; i < num_keyframes; ++i) {
    if (cur_frm->frmno == keyframes[i].frmno)
      continue;
    vector<int> kf_ftids;
    SetInliersFtids(keyframes[i], &kf_ftids);
    int pkf = CountMatchedFeatures(cur_ftids, kf_ftids);
    kfcounter.insert(make_pair(&keyframes[i], pkf));
  }

  if (kfcounter.empty()) {
    cout << "No connection"<< endl;
    return false;
}

  //Sorting and Check mathcing numbers
  vector<pair<int, Keyframe*> > vpairs;
  vpairs.reserve(num_keyframes);
  int nmax = 0;
  const int th = 10;
  Keyframe* pkfmax = NULL;
  for (map<Keyframe*, int>::iterator mit = kfcounter.begin(),
       mend = kfcounter.end(); mit!=mend; mit++) {
      //find keyframes has max number of matching
    if (mit->second > nmax) {
      nmax = mit->second;
      pkfmax = mit->first;
    }
    if (mit->second >= th) {
      vpairs.push_back(make_pair(mit->second, mit->first));
      AddConnection(mit->first, cur_frm, mit->second);
    }
  }

  if (vpairs.empty()) {
    cout << "Loop close is impossible (" << nmax <<")"<< endl;
    vpairs.push_back(make_pair(nmax, pkfmax));
    AddConnection(pkfmax, cur_frm, nmax);
    return false;
  }

  sort(vpairs.begin(), vpairs.end());
  list<Keyframe*> lkfs;
  list<int> lws;

  for (int i = 0; i < vpairs.size(); ++i) {
    lkfs.push_front(vpairs[i].second);
    lws.push_front(vpairs[i].first);
  }

  cur_frm->ordered_connectedkfs.clear();
  cur_frm->connectedkfs = kfcounter;

  if (keyframes[0].frmno == cur_frm->frmno){
    for (list<Keyframe*>::iterator it = lkfs.begin(); it != lkfs.end(); ++it) {
      if (abs((*it)->frmno - cur_frm->frmno) > 20)
        continue;
      cur_frm->ordered_connectedkfs.push_back((*it)->frmno);
    }
     cur_frm->ordered_connectedkfs.push_back(keyframes[id_].frmno);
  } else {
    for (list<Keyframe*>::iterator it = lkfs.begin(); it != lkfs.end(); ++it) {
      if (abs((*it)->frmno - cur_frm->frmno) > 20)
        continue;
      cur_frm->ordered_connectedkfs.push_back((*it)->frmno);
    }
    cur_frm->ordered_weights = vector<int>(lws.begin(), lws.end());
  }

  return true;
}

void LoopClosing::PoseGraphOptimization(vector<Keyframe>& keyframes,
                                        map<int,Vec3>& ftid_pts_map,
                                        map<Keyframe*, vector<int> > loopconnections,
                                        KeyFrameAndPose &noncorrectedsim3,
                                        KeyFrameAndPose &correctedsim3) {
  //Modeling
  ceres::Problem problem;
  int kfn = keyframes.size();
  double* sposes = new double[kfn * 7];
  double* spose = sposes;
  vector<double*> spose_map(kfn + 1);

  std::vector<Sim3,Eigen::aligned_allocator<Sim3> > 
      scws(kfn);
  std::vector<Sim3,Eigen::aligned_allocator<Sim3> > 
      corrected_swcs(kfn);


  //Set Keyframes
  for (int i = 0; i < keyframes.size(); ++i) {
    Keyframe& kf =  keyframes[i];
    const int id = i;

    if (correctedsim3.count(&kf)) {
      Sim3 s = correctedsim3[&kf];
      scws[id] = s;
      s.GetSPose(spose);
    } else {
      Mat34 pose = VisualOdometer::ToPoseMatrix(kf.pose);
      Mat3 rcw = pose.block(0,0,3,3);
      Vec3 tcw = pose.block(0,3,3,1);
      Sim3 siw(rcw, tcw, 1.0);
      scws[id] = siw;
      siw.GetSPose(spose);
    }
    spose_map[id] = spose;
    spose += 7;
  }
  

  set<std::pair<int, int> > inserted_constraints;

  Keyframe* kf = &keyframes[0];
  const int idi = kf->id;
  vector<int>& connections = kf->ordered_connectedkfs;
  Sim3 siw = scws[idi];
  Sim3 swi = siw.Inverse();
  const int idj = id_;

  Sim3 sjw = scws[idj];
  Sim3 sji = sjw * swi;
  ceres::CostFunction* cost_function = SPoseErrorFunctor::Create(sji);

  double* spose1 = spose_map[idi];
  double* spose2 = spose_map[idj];
  spose1 = spose_map[idi];

  problem.AddResidualBlock(cost_function,
      NULL,
      spose_map[idi],
      spose_map[idj]);
  inserted_constraints.insert(make_pair(min(idi, idj),
        max(idi, idj)));

  // Set normal constraints
  for (int i = 0; i < kfn; ++i) {
    Keyframe& kf = keyframes[i];
    const int idi = kf.id; 
    Sim3 swi;
    if(noncorrectedsim3.count(&kf))
      swi = noncorrectedsim3[&kf].Inverse();
    else
      swi = scws[idi].Inverse();

    // Covisible edges
    vector<int> covisible_kfs = kf.ordered_connectedkfs;
    for (int j = 0; j < covisible_kfs.size(); ++j) {
      int frmno = covisible_kfs[j];
      int idx = FindKeyframe(frmno, keyframes);
      Keyframe* vkf = &keyframes[idx];
        const int idv = vkf->id;
        if (inserted_constraints.count(std::make_pair(
                std::min(idi, idv), std::max(idi, idv))) || idi == idv) {
          continue;
        }
        Sim3 svw;
        if (noncorrectedsim3.count(vkf))
          svw = noncorrectedsim3[vkf];
        else
          svw = scws[idv];
        Sim3 svi = svw * swi;
        ceres::CostFunction* cost_function = SPoseErrorFunctor::Create(svi);
        problem.AddResidualBlock(cost_function,
                                 NULL,
                                 spose_map[idi],
                                 spose_map[idv]);
    }
  }

  if(spose_map[matchedkf_->id])
    problem.SetParameterBlockConstant(spose_map[matchedkf_->id]);

  // Solving
  ceres::Solver::Options options;
  options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.max_num_iterations = 20;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  // Sim3 -> SE3
  for (int i = 0; i < kfn; ++i) {
    Keyframe& kf = keyframes[i];
    int id = kf.id;
    spose = spose_map[id];
    keyframes[i].pose = GetPoseFromSim3(spose);
    Sim3 s(spose);
    corrected_swcs[id] = s.Inverse();
  }

  map<int, Vec3> new_ftids_pts_map;
  //Refine 3D Points
  int ft;
  int kfidx = 0;
  for (map<int, Vec3>::iterator it = ftid_pts_map.begin(); it != ftid_pts_map.end(); 
      it++) {
    Sim3 pose, pose_corrected;
    const int ftid = it->first;
    for (int i = 0; i < keyframes.size(); ++i) {
      const vector<int>& ftids = keyframes[i].ftids;
      bool find = false;
      for (int j = 0; j < ftids.size(); ++j) {
        if (ftid == ftids[j]) {
          find = true;
          pose_corrected = corrected_swcs[i];
          pose = scws[i];
          break;
        }
      }
      if (find)
        break;
    }
    Vec3 pts_ori = it->second;
    Vec3 pts_new = pose_corrected.Transform(pose.Transform(pts_ori));
    new_ftids_pts_map.insert(make_pair(ftid, pts_new));
  }

  ftid_pts_map.swap(new_ftids_pts_map);

  LOG(INFO) << summary.FullReport();
  cout << "Loop closed !" << endl << endl;

  delete[] sposes;
}


void SetKeyframeID(vector<Keyframe>& keyframes) {
  for (int i = 0;i < keyframes.size(); ++i) {
    keyframes[i].id = i;
  }
}

Vec6 GetPoseFromSim3(const double* spose) {
  Mat34 pose_mat;
  Vec6 pose;
  double s = spose[6];
  double r[9];
  ceres::AngleAxisToRotationMatrix(spose, r);
  for (int i = 0; i < 3; ++i) {
    pose_mat(i, 3) = spose[i + 3] / s;
    for (int j = 0; j < 3; ++j)
      pose_mat(j, i) = r[i * 3 + j];
  }
  pose = VisualOdometer::ToPoseVector(pose_mat);
  
  return pose;
}







}
