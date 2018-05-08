// optimize.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// Author: Euntae Hong(dragon1301@naver.com)

#include <map>
#include <set>
#include <list>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "optimize.h"
#include "rvslam_common.h"
#include "rvslam_util.h"
#include "triangulation.h"
#include "optimize.h"

#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;

DEFINE_int32(num_keyframe, 6, "Number of keyframes to keep.");
DEFINE_double(ceres_huber_loss_sigma, 1,
             "The sigma parameter of Huber loss function.");
DEFINE_int32(ceres_num_iterations, 50, "number of iterations.");
DEFINE_double(ceres_max_solver_time, 1, "max time of optimization");
DEFINE_double(pose_inlier_thr2, 7.0, "Pose inlier threshold in pixels.");
DEFINE_double(pose_smoothness_scale2, 0.30,
              "Pose smoothness scale in bundle adjustment.");

const int kNumKeyframes = 5000;

using namespace std;

namespace rvslam {

struct SmoothMotionConstraint {
  SmoothMotionConstraint(double scale, int idx) : scale(scale), idx(idx) {}

  template <typename T>
  bool operator()(const T* const pose0,
                  const T* const pose1,
                  T* residual) const {
    T pR0[9], pR1[9], pR0R1t[9];  // R0 and R1 are column-major matrices.
    ceres::AngleAxisToRotationMatrix(pose0, pR0);
    ceres::AngleAxisToRotationMatrix(pose1, pR1);
    ceres::MatrixAdapter<T, 1, 3> R0 = ceres::ColumnMajorAdapter3x3(pR0);
    ceres::MatrixAdapter<T, 3, 1> R1t = ceres::RowMajorAdapter3x3(pR1);
    ceres::MatrixAdapter<T, 1, 3> R0R1t = ceres::ColumnMajorAdapter3x3(pR0R1t);
    for (int c = 0; c < 3; ++c) {
      for (int r = 0; r < 3; ++r) {
        R0R1t(r, c) = T(0);
        for (int i = 0; i < 3; ++i) R0R1t(r, c) += R0(r, i) * R1t(i, c);
      }
    }
    const T* pt0 = &pose0[3];
    const T* pt1 = &pose1[3];
    T pc1[3];
    for (int r = 0; r < 3; ++r) {
      pc1[r] = pt0[r] - (R0R1t(r, 0) * pt1[0] + R0R1t(r, 1) * pt1[1] +
                         R0R1t(r, 2) * pt1[2]);
    }
    T pz1[3] = { R0R1t(0, 2), R0R1t(1, 2), R0R1t(2, 2) };

    T dot = pz1[0] * pc1[0] + pz1[1] * pc1[1] + pz1[2] * pc1[2];
    residual[0] = scale * (pc1[0] - dot * pz1[0]);
    residual[1] = scale * (pc1[1] - dot * pz1[1]);
    residual[2] = scale * (pc1[2] - dot * pz1[2]);
    return true;
  }

  double scale;
  int idx;
};

struct PoseAlignError {
  PoseAlignError(const double pose_ref[6]) {
    for (int i = 0; i < 6; ++i) pose_[i] = pose_ref[i];
  }

  template <typename T>
  bool operator()(const T* const pose, T* residual) const {
    for (int i = 0; i < 3; ++i) residual[i] = (pose[i] - pose_[i]) * rot_coef_;
    for (int i = 4; i < 6; ++i) residual[i] = (pose[i] - pose_[i]) * tr_coef_;
    return true;
  }

  double pose_[6];
  static double rot_coef_, tr_coef_;
};

double PoseAlignError::rot_coef_ = 1.0;
double PoseAlignError::tr_coef_ = 1.0;

struct ReprojectionErrorMonocular {
  ReprojectionErrorMonocular (double x1, double y1)
    : x1(x1), y1(y1) {}

  template <typename T>
    bool operator()(const T* const pose,
        const T* const point,
        T* residual) const {
      T p[3];
      ceres::AngleAxisRotatePoint(pose, point, p);
      p[0] += pose[3];
      p[1] += pose[4];
      p[2] += pose[5];

      T nx = p[0] / p[2];
      T ny = p[1] / p[2];

      residual[0] = nx - T(x1);
      residual[1] = ny - T(y1);

      return true;
    }

  double x1, y1;
};

Optimize::Optimize(const char *voc_file, VisualOdometer::Calib calib,
                   Map* world)
  : world_(world), is_keyframe_(&world->is_keyframe_) {
    const double f = (calib.fx + calib.fy) / 2;
    pose_inlier_thr_ = FLAGS_pose_inlier_thr2 / f;
    loop.Setup(voc_file, calib.fx, calib.fy, calib.cx, calib.cy,
               pose_inlier_thr_);
  }

bool Optimize::Process() {
  Keyframe& cur_frm = world_->keyframes_[0];
  vector<int>& ftids = cur_frm.ftids;
  Mat3X& normalized_pts = cur_frm.normalized_pts;

  // Optimize the keyframe poses and world points.
  ProfileBegin("232.optimize", &pdb_);
  LocalOptimizeWorld();
  ProfileEnd("232.optimize", &pdb_);

  bool lp = false;
  ProfileBegin("233.loopclose", &pdb_);
  lp = loop.Process(cur_frm, world_->keyframes_, world_->ftid_pts_map_);
  //lp = false;
  ProfileEnd("233.loopclose", &pdb_);
  world_->is_loopclosed_ = lp;

  if (lp) {
    // Optimize the keyframe poses and world points.
    LocalOptimizeWorld();
    // Update pose inliers in each keyframes.
  }

  // Update pose inliers in each keyframes.
  Mat3X empty_world_pts;
  int num_iter = 2;
  if (num_iter > world_->keyframes_.size())
    num_iter = world_->keyframes_.size();
  for (int i = 0; i < num_iter; ++i) {
    Keyframe& kf = world_->keyframes_[i];
    FindPoseInliers(kf.ftids, kf.normalized_pts, empty_world_pts, kf.pose,
        pose_inlier_thr_ / 3.0, &kf.pose_inliers);
  }

  if (world_->keyframes_.size() <= kNumKeyframes) world_->keyframes_.push_back(Keyframe());
  for (int i = world_->keyframes_.size() - 1; i > 1; --i) {
    world_->keyframes_[i].Swap(&world_->keyframes_[i - 1]);
  }
  world_->keyframes_[1] = cur_frm;

  ProfileEnd("23.kfrm", &pdb_);

  world_->pose_inliers_ = cur_frm.pose_inliers;
  for (int i = 0; i < world_->pose_inliers_.size(); ++i) {
    world_->pose_inliers_[i] = ftids[cur_frm.pose_inliers[i]];
  }
}

void Optimize::LocalOptimizeWorld() {
  if (world_->keyframes_.size() < 3 || world_->ftid_pts_map_.size() < 1) return;
  if (world_->keyframes_.size() == 2 &&
      world_->keyframes_[0].frmno == world_->keyframes_[1].frmno) return;

  ceres::Problem problem;
  //Local Bundle Adjustment
  int num_iter = FLAGS_num_keyframe;
  if (num_iter > world_->keyframes_.size())
    num_iter = world_->keyframes_.size();
  set<int> ftids;
  for (int kfidx = 0; kfidx < num_iter; ++kfidx) {
    Keyframe& kf = world_->keyframes_[kfidx];
    bool is_fixed = (kfidx > num_iter - 3);
    int cnt = 0;
    for (int i = 0; i < kf.pose_inliers.size(); ++i) {
      const int idx = kf.pose_inliers[i];
      const int ftid = kf.ftids[idx];

      if (!is_fixed) {
        ftids.insert(ftid);
      } else {
        if (ftids.count(ftid) <= 0) continue;
      }

      map<int, Vec3>::iterator it = world_->ftid_pts_map_.find(ftid);
      if (it == world_->ftid_pts_map_.end()) {
        LOG(FATAL) << "ftid " << ftid << " not found.";
        continue;
      }
      Vec3& pt = it->second;
      ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<ReprojectionErrorMonocular, 2, 6, 3>(
            new ReprojectionErrorMonocular(
              kf.normalized_pts(0, idx), kf.normalized_pts(1, idx)));
      ceres::LossFunction* loss_function =
        new ceres::HuberLoss(FLAGS_ceres_huber_loss_sigma);

      problem.AddResidualBlock(cost_function, loss_function,
          kf.pose.data(), pt.data());
      cnt++;
    }

    if (!cnt)
      continue;

    //fix scale
    if (is_fixed) 
      problem.SetParameterBlockConstant(kf.pose.data()); 
  }

  ceres::Solver::Options options;
  options.max_num_iterations = FLAGS_ceres_num_iterations;
  options.max_solver_time_in_seconds = FLAGS_ceres_max_solver_time;
  options.gradient_tolerance = 1e-9;
  options.function_tolerance = 1e-9;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << summary.BriefReport();
  //LOG(INFO) << summary.FullReport();
  lba_ftids_ = ftids;
}

void Optimize::FindPoseInliers(const vector<int>& ftids,
    const Mat3X& normalized_pts,
    const Mat3X& world_pts,
    const Vec6& pose,
    const double inlier_thr,
    vector<int>* inliers) const {
  CHECK_NOTNULL(inliers);
  CHECK_EQ(ftids.size(), normalized_pts.cols());
  const Mat34 pose_mat = VisualOdometer::ToPoseMatrix(pose);
  const bool world_pts_given = (world_pts.size() == ftids.size());
  const double inlier_thr_squared = inlier_thr * inlier_thr;
  inliers->clear();
  inliers->reserve(ftids.size());
  for (int i = 0; i < ftids.size(); ++i) {
    Vec3 world_pt;
    map<int, Vec3>::const_iterator it;
    if (world_pts_given) {
      world_pt = world_pts.col(i);
    } else if ((it = world_->ftid_pts_map_.find(ftids[i])) != world_->ftid_pts_map_.end()) {
      world_pt = it->second;
    } else {
      continue;
    }
    Vec3 x = pose_mat * Hom(world_pt);
    Vec2 x0 = normalized_pts.block(0, i, 2, 1);
    Vec3 diff = UnitVector(x) - UnitVector(Hom(x0));
    //    Vec2 diff = ProjectEuc(pose_mat, world_pt) - stereo_pts.block(0, i, 2, 1);
    double err = diff.squaredNorm();
    if (err < inlier_thr_squared) inliers->push_back(i);
  }
}

void Optimize::UpdateWorldPoints() {
  int end = world_->keyframes_.size()-1;
  Keyframe& kf_cur = world_->keyframes_[0];
  Keyframe& kf_rel = world_->keyframes_[1];
  // Estimate new 3D points
  const Mat3X& pts1 = kf_rel.normalized_pts;
  const Mat3X& pts2 = kf_cur.normalized_pts;
  const vector<int>& ftids1 = kf_rel.ftids;
  const vector<int>& ftids2 = kf_cur.ftids;
  Mat3X new_pts1, new_pts2;
  vector<int> new_ftids;
  // Find correspondance between 2 frame's features
  FindMatchedPoints(pts1, ftids1, pts2, ftids2, &new_pts1, &new_pts2, &new_ftids);
  const Mat34 pose_cur = VisualOdometer::ToPoseMatrix(kf_cur.pose);
  const Mat34  pose_rel = VisualOdometer::ToPoseMatrix(kf_rel.pose);
  // Triangulation
  Eigen::ArrayXXd pts_4d(4, new_ftids.size());
  TriangulatePoints(pose_rel, pose_cur, new_pts1, new_pts2, &pts_4d); 
  pts_4d.rowwise() /= pts_4d.row(3);
  Mat34 cur_pose = VisualOdometer::ToPoseMatrix(kf_cur.pose);
  Mat3X pts3d = cur_pose * pts_4d.matrix();
  const int origin_map = world_->ftid_pts_map_.size();

  /*
     for (int i = 0; i < new_ftids.size(); ++i) {
     const int ftid = new_ftids[i];
     map<int, Vec3>::const_iterator it = world_->ftid_pts_map_.find(ftid);
     Vec3 pos;
     if (it == world_->ftid_pts_map_.end()) {
     pos = pts_4d.col(i).block(0,0,3,1);
  //delete -z
  if (pos(2) < 0)
  continue;
  }
  world_->ftid_pts_map_.insert(make_pair(ftid, pos));
  }
  */


  // Copy only visible world points.
  map<int, Vec3> new_ftid_pts_map;
  for (int i = 0; i < world_->keyframes_.size(); ++i) {
    const Keyframe& kf = world_->keyframes_[i];
    for (int j = 0; j < kf.ftids.size(); ++j) {
      bool find = false;
      const int ftid = kf.ftids[j];
      map<int, Vec3>::const_iterator it = world_->ftid_pts_map_.find(ftid);
      Vec3 pos;
      // Add new points
      if (it == world_->ftid_pts_map_.end()) {
        //  const int idx = FindInSortedArray(new_ftids, ftid);
        int idx = -1;
        for (int j = 0; j < new_ftids.size(); ++j){
          if (ftid == new_ftids[j])
            idx = j;
        }
        if (idx < 0)
          continue;
        pos = pts_4d.col(idx).block(0,0,3,1);
        const Vec3 rel_pos = pts3d.col(idx);
        //delete -z
        if (rel_pos(2) < 0)
          continue;
        find = true;
        // Add a newly found world point.
      } else {
        find = true;
        pos = it->second;
      }
      if (find)
        new_ftid_pts_map.insert(make_pair(ftid, pos));
    }
  }

  world_->ftid_pts_map_.swap(new_ftid_pts_map);
}

void Optimize::FindMatchedPoints(const Mat3X& pts1, const vector<int>& ftids1,
    const Mat3X& pts2, const vector<int>& ftids2,
    Mat3X* new_pts1, Mat3X* new_pts2,
    vector<int>* new_ftids) const {
  new_ftids->clear();
  const int n = pts1.cols() > pts2.cols() ? pts1.cols() : pts2.cols();
  new_pts1->resize(3,n);
  new_pts2->resize(3,n);
  new_ftids->reserve(n);

  //Find correspondance points on each 2 frames 
  for (int i = 0; i < ftids1.size(); ++i) {
    const int ftid = ftids1[i];
    int idx = -1;
    for (int j = 0; j < ftids2.size(); ++j){
      if (ftid == ftids2[j])
        idx = j;
    }
    //const int idx = FindInSortedArray(ftids2, ftid);
    if (idx < 0)
      continue;
    const Vec3& p1 = pts1.col(i);
    const Vec3& p2 = pts2.col(idx);
    const int col_idx = new_ftids->size();
    new_pts1->col(col_idx) = p1;
    new_pts2->col(col_idx) = p2;
    new_ftids->push_back(ftid);
  }
  new_pts1->conservativeResize(3, new_ftids->size());
  new_pts2->conservativeResize(3, new_ftids->size());
}


}
