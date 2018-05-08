// visual_odometer.cc
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
#include "rvmap.h"

#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;

//#define USE_QUATERNION

using namespace std;

const int kNumKeyframes = 5000;
DEFINE_int32(num_keyframes, 6, "Number of keyframes to keep.");
DEFINE_double(pose_inlier_thr, 1.0, "Pose inlier threshold in pixels.");
DEFINE_int32(pose_min_inliers, 10,
             "Minimum number of inlier points in pose estimation.");
DEFINE_double(keyframe_translation_thr, 1,
              "Translation threshold for making a new keyframe (in meters).");
DEFINE_double(pose_smoothness_scale, 1.0,
              "Pose smoothness scale in bundle adjustment.");
DEFINE_int32(covisible_thr, 50, "number of matched feature for covisible graph.");
DEFINE_int32(initial_matching_thr, 50, "number of matched feature for initialization.");
DEFINE_double(keyframe_rotation_thr, 5,
              "Rotation threshold for a new keyframe (in degrees).");
DEFINE_double(overlap_th, 0.4, "overlap ratio threshold when we select KF");

namespace rvslam {

namespace {

ostream& operator<<(ostream& os, const VisualOdometer::Calib& c) {
  return os << c.fx << ", " << c.fy << " / " << c.cx << ", " << c.cy;
}

template <typename T, int nc, int nr>
istream& operator>>(istream& is, Eigen::Matrix<T, nc, nr>& mat) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) is >> mat(i, j);
  }
  return is;
}

struct ReprojectionErrorStereo {
  ReprojectionErrorStereo(double x1, double y1, double x2, double y2)
      : x1(x1), y1(y1), x2(x2), y2(y2) {}

  template <typename T>
  bool operator()(const T* const pose,
                  const T* const point,
                  T* residual) const {
    // Point: Xl = R * P + t, Xr = Xl;  Xr[0] += baseline;
    T p[3];
    ceres::AngleAxisRotatePoint(pose, point, p);
    p[0] += pose[3];
    p[1] += pose[4];
    p[2] += pose[5];

    residual[0] = p[0] / p[2] - T(x1);
    residual[1] = p[1] / p[2] - T(y1);
    residual[2] = (p[0] + baseline) / p[2] - T(x2);
    residual[3] = residual[1];
    return true;
  }

  double x1, y1, x2, y2;
  static double baseline;
};

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

double ReprojectionErrorStereo::baseline = 1.0;

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

}  // namespace

Vec6 VisualOdometer::ToPoseVector(const Mat34& pose_mat) {
  Vec6 pose;
  ceres::RotationMatrixToAngleAxis(pose_mat.data(), pose.data());
  pose.segment(3, 3) = pose_mat.block(0, 3, 3, 1);
  return pose;
}

Mat34 VisualOdometer::ToPoseMatrix(const Vec6& pose) {
  Mat34 pose_mat;
  ceres::AngleAxisToRotationMatrix(pose.data(), pose_mat.data());
  pose_mat.block(0, 3, 3, 1) = pose.segment(3, 3);
  return pose_mat;
}

Mat34 VisualOdometer::InversePoseMatrix(const Mat34& pose_mat) {
  Mat3 Rt = pose_mat.block(0, 0, 3, 3).transpose();
  Mat34 pose_mat_inv;
  pose_mat_inv.block(0, 0, 3, 3) = Rt;
  pose_mat_inv.block(0, 3, 3, 1) = -Rt * pose_mat.block(0, 3, 3, 1);
  return pose_mat_inv;
}

Vec6 VisualOdometer::RelativePoseVector(const Vec6& ref_pose,
                                        const Vec6& pose) {
  return ToPoseVector(RelativeTransform(ToPoseMatrix(ref_pose),
                                        ToPoseMatrix(pose)));
}


//-----------------------------------------------------------------------------

VisualOdometer::VisualOdometer(const Calib& calib,
                               Map* world,
                               const char* voc_file)
    : calib_(calib), world_(world) {
  LOG(INFO) << "calib: " << calib_;
  world_->keyframes_.reserve(kNumKeyframes + 1);
  world_->keyframes_.resize(1);
  const double f = (calib.fx + calib.fy) / 2;
  pose_inlier_thr_ = FLAGS_pose_inlier_thr / f;
}

bool VisualOdometer::ProcessFrame(int frmno,
                                  const std::vector<int>& ftids,
                                  const Mat3X& image_pts,
                                  cv::Mat& descriptors) {
  CHECK_EQ(ftids.size(), image_pts.cols());

  Keyframe& cur_frm = world_->keyframes_[0];
  cur_frm.frmno = frmno;
  cur_frm.ftids = ftids;
  cur_frm.descriptors = descriptors;
  world_->is_keyframe_ = true;
  world_->is_success_ = true;

  // Convert image_pts into normalized pts.
  Mat3X& normalized_pts = cur_frm.normalized_pts;
  MakeNormalizedPoints(image_pts, &normalized_pts);

  // Estimate the current frame's pose and pose-inliers.
  if (world_->keyframes_.size() <= 1) {
    // First frame : initialize pose, pose_inliers.
    cur_frm.pose.fill(0.0);
    cur_frm.pose_inliers.resize(ftids.size());
    for (int i = 0; i < ftids.size(); ++i) cur_frm.pose_inliers[i] = i;
    world_->is_loopclosed_ = false;
  } else {
    ProfileBegin("21.pose", &pdb_);

    // Only for 2 frames : initialize map using 5points and triangulation.
    if (world_->keyframes_.size() <= 2 &&
       (cur_frm.frmno - world_->keyframes_[1].frmno) > 10) {
      const Mat3X& pts1 = world_->keyframes_[1].normalized_pts;
      const Mat3X& pts2 = normalized_pts;
      const vector<int>& ftids1 = world_->keyframes_[1].ftids;
      const vector<int>& ftids2 = cur_frm.ftids;
      Mat3X new_pts1, new_pts2;
      vector<int> new_ftids;
      //Find correspondance between 2 frame's features
      FindMatchedPoints(pts1, ftids1, pts2, ftids2,
                        &new_pts1, &new_pts2, &new_ftids);
      // Check number of matchin points
      if (new_pts1.cols() < FLAGS_initial_matching_thr) {
        world_->keyframes_[1] = cur_frm;
        LOG(INFO) << "Resetup origin keyframe(" << new_pts1.cols()
                  << ")" << endl;
        world_->is_success_ = false;
        return false;
      }
        
      if (!InitializeWorld(new_pts1, new_pts2, new_ftids)){
        world_->is_success_ = false;
        return false;
      }
    }

    // Compute the 3D pose and find inliers.
    Mat3X world_pts;
    vector<int> ftid_idx_found;
    BuildWorldPointsMatrixForP3P(ftids, &ftid_idx_found, &world_pts);

    Mat3X normalized_image_pts(3, ftid_idx_found.size());
    for (int i = 0; i < ftid_idx_found.size(); ++i) {
      const int idx = ftid_idx_found[i];
      normalized_image_pts.col(i)
        << normalized_pts(0, idx), normalized_pts(1, idx), 1.0;
    }

    // Filter points not consistent with vehicle motion.
    vector<int> ftid_idx_found_org = ftid_idx_found;
    Mat3X world_pts_org = world_pts;
    Mat3X normalized_image_pts_org = normalized_image_pts;

    // Perform P3P RANSAC.
    Mat34 pose_mat;
    vector<int> pose_inliers;
    if (!RobustEstimatePoseP3P(normalized_image_pts, world_pts,
          pose_inlier_thr_, &pose_mat, &pose_inliers,
          NULL, 1e-4)) {
      LOG(ERROR) << "RobustEstimatePosesP3P failed: " << world_pts.cols()
                 << " pts, thr=" << pose_inlier_thr_ << " - skipping the frame.";
      world_->is_success_ = false;
      return false;
    }
    cur_frm.pose = ToPoseVector(pose_mat);
    cur_frm.pose_inliers.resize(pose_inliers.size());
    for (int i = 0; i < pose_inliers.size(); ++i) {
      cur_frm.pose_inliers[i] = ftid_idx_found[pose_inliers[i]];
    }
    GetFeatureProjections(ftids, cur_frm.pose, &p3p_fpos);
    ProfileEnd("21.pose", &pdb_);

    ftid_idx_found.swap(ftid_idx_found_org);
    world_pts.swap(world_pts_org);
    normalized_image_pts.swap(normalized_image_pts_org);

    if(!PoseOptimize())
      LOG(INFO) << "Not optimized" << endl;
    FindPoseInliers(ftids, normalized_pts, world_pts, cur_frm.pose,
        pose_inlier_thr_, &cur_frm.pose_inliers);
    LOG(INFO) << setfill(' ') << "pose_inliers: " << cur_frm.pose_inliers.size()
              << ", pose: " << cur_frm.pose.transpose()
              << " (" << world_->keyframes_.size() << " kfs, " << world_->ftid_pts_map_.size() << "pts)";

    // Determine if the current frame is a keyframe.
    const Keyframe& last_kf = world_->keyframes_[1];
    {
      const vector<int>& ftids1 = world_->keyframes_[1].ftids;
      const vector<int>& ftids2 = cur_frm.ftids;
      if (!CheckKeyframe(ftids1, ftids2))
        world_->is_keyframe_ = false;
    }
  }

  if (world_->is_keyframe_ == false) {
  } else { 
    ProfileBegin("23.kfrm", &pdb_);
    if (world_->keyframes_.size() == 1) {
      if (world_->keyframes_.size() <= kNumKeyframes) world_->keyframes_.push_back(Keyframe());
      for (int i = world_->keyframes_.size() - 1; i > 1; --i) {
        world_->keyframes_[i].Swap(&world_->keyframes_[i - 1]);
      }
      world_->keyframes_[1] = cur_frm;
      world_->is_success_ = false;
      return false;
    }

    //Discard invisible world points and add new world points.
    ProfileBegin("231.pts", &pdb_);
    if (world_->keyframes_.size() > 2)
      UpdateWorldPoints();
    ProfileEnd("231.pts", &pdb_);
    Mat3X empty_world_pts;

    // Update pose inliers in each keyframes.
    int num_iter = 2;
    if (num_iter > world_->keyframes_.size())
      num_iter = world_->keyframes_.size();
    for (int i = 0; i < num_iter; ++i) {
      Keyframe& kf = world_->keyframes_[i];
      FindPoseInliers(kf.ftids, kf.normalized_pts, empty_world_pts, kf.pose,
          pose_inlier_thr_, &kf.pose_inliers);
    }
  }

//  world_->pose_inliers_ = cur_frm.pose_inliers;
  return true;
}

void VisualOdometer::MakeNormalizedPoints(const Mat3X& image_pts,
                                          Mat3X* normalized_pts) const {
  CHECK_NOTNULL(normalized_pts);
  const double fx = calib_.fx, fy = calib_.fy, cx = calib_.cx, cy = calib_.cy;
  const double k1 = calib_.k0, k2 = calib_.k1, k4 = 0;
  normalized_pts->resize(3, image_pts.cols());
  for (int i = 0; i < image_pts.cols(); ++i) {
    double nx = (image_pts(0, i) - cx) / fx;
    double ny = (image_pts(1, i) - cy) / fy;
    double nx_out, ny_out;
    
    if (k1 != 0.0 || k2 != 0.0) {
      double x = nx, y = ny;
      for (int i = 0; i < 10; i++) {
        const double x2 = pow(x,2), y2 = pow(y,2), xy = 2 * x * y, r2 = x2 + y2;
        const double rad = 1 + r2 * (k1 + (r2 * k2));
        const double ux = nx / rad;
        const double uy = ny / rad;
        const double dx = x - ux, dy = y - uy;
        x = ux,  y = uy;
        if (pow(dx,2) + pow(dy,2) < 1e-9) break;
      }
      nx = x, ny = y;
    }
    nx_out = nx, ny_out = ny;
    normalized_pts->col(i) << nx_out, ny_out, 1;
  }
}

bool VisualOdometer::InitializeWorld(const Mat3X& pts1, const Mat3X& pts2,
                                     const vector<int>& ftids) {
  world_->ftid_pts_map_.clear();
  Mat34 rt_rel;
  Mat34 rt_origin; 
  vector<int> inliers;
  double score;

  // Check purerotation
  Mat34 rt_rel_rot;
  Mat34 rt_origin_rot; 
  vector<int> inliers_rot;
  double score_rot;
  RobustEstimateRotationMat(pts1, pts2, &rt_rel_rot, &inliers_rot, 
      &score_rot, pose_inlier_thr_);
  double inlier_ratio_rot = (double)inliers_rot.size() / pts1.cols();

  if (inlier_ratio_rot > 0.6) {
    cout << "Not initialized from rotation" << endl ;
    return false;
  }

  //Estimate Relative pose between origin and relative frame using 5 points
  RobustEstimateRelativePose5pt(pts1, pts2, &rt_rel, &inliers, 
                                &score, pose_inlier_thr_);
  double inlier_ratio = (double)inliers.size() / pts1.cols();
  rt_origin.fill(0.0);
  rt_origin(0,0) = rt_origin(1,1) = rt_origin(2,2) = 1; 

  if (inlier_ratio < 0.6) {
    cout << "Not initialized(" <<inlier_ratio << ", " << pts1.cols() << ")" << endl  ;
    return false;
  }
  LOG(INFO) << "inlier : " << inlier_ratio_rot << " " << inlier_ratio << endl;

  const int n = inliers.size();
  Mat3X new_pts1(3, n), new_pts2(3 ,n);
  vector<int> new_ftids;
  for (int i = 0; i < n; ++i) {
    const int idx = inliers[i];
    new_pts1.col(i) = pts1.col(idx); 
    new_pts2.col(i) = pts2.col(idx); 
    new_ftids.push_back(ftids[idx]);
  }


  //Estimate 3D points of the points set
  Eigen::ArrayXXd pts_4d(4, n);
  TriangulatePoints(rt_origin, rt_rel, new_pts1, new_pts2, &pts_4d); 

  //Add on the map
  for (int i = 0; i < n; ++i) {
    const int ftid = new_ftids[i];
    Vec3 pos;
    pos << pts_4d(0, i), pts_4d(1, i), pts_4d(2, i);
    if (pos(2) < 0.0)  {
        continue;
    }
    world_->ftid_pts_map_.insert(make_pair(ftid, pos));
  }

  //Check reprojection Error
  Mat3X pts(3,inliers.size());
  vector<int> ftids_new;
  for (int i = 0; i < inliers.size(); ++i) {
    const int idx = inliers[i];
    pts.col(i) = pts1.col(idx);
    ftids_new.push_back(ftids[idx]);
  }
  vector<int> pose_inliers;
  Vec6 cur_frm = ToPoseVector(rt_origin);
  Mat3X empty_world_pts;
  FindPoseInliers(ftids_new, pts, empty_world_pts, cur_frm,
                  pose_inlier_thr_, &pose_inliers);

  LOG(INFO)  << "IntializeWorld: pose_inliers =" << pose_inliers.size() << endl;

  if (pose_inliers.size() < 10) {
    LOG(INFO)  << "Not initialized " << endl ;
    return false;
  }
  

  return true;
}


void VisualOdometer::BuildWorldPointsMatrixForP3P(const vector<int>& ftids,
                                                  vector<int>* ftid_idx_found,
                                                  Mat3X* world_pts) const {
  CHECK_NOTNULL(ftid_idx_found);
  CHECK_NOTNULL(world_pts);
  // Initialize ftid_idx_found, world_pts, and normalized_image_pts.
  ftid_idx_found->clear();
  if (world_->ftid_pts_map_.empty()) return;
  const int num_ftids = ftids.size();
  ftid_idx_found->reserve(num_ftids);
  world_pts->resize(3, num_ftids);
  // Find image_pts that has matched world_pts in ftid_pts_map_.
  for (int i = 0; i < ftids.size(); ++i) {
    map<int, Vec3>::const_iterator it = world_->ftid_pts_map_.find(ftids[i]);
    if (it == world_->ftid_pts_map_.end()) continue;
    const int col_idx = ftid_idx_found->size();
    ftid_idx_found->push_back(i);
    world_pts->col(col_idx) = it->second;
  }
  world_pts->conservativeResize(3, ftid_idx_found->size());
}

void VisualOdometer::FindPoseInliers(const vector<int>& ftids,
                                     const Mat3X& normalized_pts,
                                     const Mat3X& world_pts,
                                     const Vec6& pose,
                                     const double inlier_thr,
                                     vector<int>* inliers) const {
  CHECK_NOTNULL(inliers);
  CHECK_EQ(ftids.size(), normalized_pts.cols());
  const Mat34 pose_mat = ToPoseMatrix(pose);
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
    double err = diff.squaredNorm();
    if (err < inlier_thr_squared) inliers->push_back(i);
  }
}

void VisualOdometer::FindMatchedPoints(const Mat3X& pts1, const vector<int>& ftids1,
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

Vec3 VisualOdometer::EstimateRotationalVelocity() const {
  Vec3 w;
  w << 0, 0, 0;
  int denom = 0;
  const Keyframe& cur_frm = world_->keyframes_[0];
  for (int i = 1; i < world_->keyframes_.size(); ++i) {
    const Keyframe& kf = world_->keyframes_[i];
    if (cur_frm.frmno == world_->keyframes_[i].frmno) continue;
    Vec6 rel = RelativePoseVector(cur_frm.pose, kf.pose);
    int frmno_diff = kf.frmno - cur_frm.frmno;
    w += rel.segment(0, 3);
    denom += frmno_diff;
  }
  if (denom != 0) w /= denom;
  return w;
}

//Input pts : normalized pts
void VisualOdometer::GetUndistortedPoints(const Mat3X& pts, Mat3X* pts_out)
                                           const {
  const double fx = calib_.fx, fy = calib_.fy, k1 = calib_.k2, k2 = calib_.k1;
  const double cx = calib_.cx, cy = calib_.cy;

  const int n = pts.cols();
  pts_out->resize(3, n);
  for (int i = 0; i < n; ++i){
    Vec3 p;
    p << pts(0,i), pts(1,i), pts(2,i);
    const double nx = p(0); 
    const double ny = p(1);
    const double r2 = nx * nx + ny * ny;
    const double distortion = 1.0 + r2 * (k1 + k2 * r2);
    p(0) = distortion * nx;
    p(1) = distortion * ny;
    pts_out->col(i) << p(0), p(1), p(2);
  }
}

void VisualOdometer::GetFeatureProjections(const vector<int>& ftids,
                                           const Vec6& pose,
                                           Mat2X* pos) const {
  CHECK_NOTNULL(pos);
  const Keyframe& cur_frm = world_->keyframes_[0];
  const double fx = calib_.fx, fy = calib_.fy, cx = calib_.cx, cy = calib_.cy;
  const Mat34 pose_mat = ToPoseMatrix(pose);
  pos->resize(2, ftids.size());
  pos->fill(-1.0);
  for (int i = 0; i < ftids.size(); ++i) {
    map<int, Vec3>::const_iterator it;
    if ((it = world_->ftid_pts_map_.find(ftids[i])) == world_->ftid_pts_map_.end()) continue;
    const Vec3& world_pt = it->second;
    Vec2 x = ProjectEuc(pose_mat, world_pt);
    pos->coeffRef(0, i) = x[0] * fx + cx;
    pos->coeffRef(1, i) = x[1] * fy + cy;
  }
}

void VisualOdometer::UpdateWorldPoints() {
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


  /*
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
  */

  //LOG(INFO) << "Map is Updated : " << origin_map << "->"
   // << world_->ftid_pts_map_.size() << endl<<endl;
  //world_->ftid_pts_map_.swap(new_ftid_pts_map);
}

bool VisualOdometer::PoseOptimize() {
  if (world_->keyframes_.size() < 3 || world_->ftid_pts_map_.size() < 1) return false;
  if (world_->keyframes_.size() == 2 &&
      world_->keyframes_[0].frmno == world_->keyframes_[1].frmno) return false;

  ceres::Problem problem;
  int num_iter = 10;
  if (num_iter > world_->keyframes_.size())
    num_iter = world_->keyframes_.size();

  set<int> ftids;
  for (int kfidx = 0; kfidx < num_iter; ++kfidx) {
    Keyframe& kf = world_->keyframes_[kfidx];
    bool is_fixed = (kfidx > 0);
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
        new ceres::HuberLoss(1);

      problem.AddResidualBlock(cost_function, loss_function,
          kf.pose.data(), pt.data());
      cnt++;
    }

    if (!cnt)
      continue;

    //fix the other keyframes
    if (kfidx > 0) {
      problem.SetParameterBlockConstant(kf.pose.data()); 
    }
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 50;
    options.gradient_tolerance = 1e-9;
    options.function_tolerance = 1e-9;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  return true;
  LOG(INFO) << summary.BriefReport();
}


bool VisualOdometer::CheckKeyframe(const vector<int>& ftids1, const vector<int>& ftids2) const {
  //initialize
  if (world_->keyframes_.size() < 3)
    return true;
  //Check 1. 
  const int last_kf = world_->keyframes_[1].frmno;
  const int cur_kf = world_->keyframes_[0].frmno;

  if (cur_kf - last_kf < 3)
    return false;

  const int t = world_->keyframes_.size();
  Vec6 compare_pose = VisualOdometer::RelativePoseVector(world_->keyframes_[t-1].pose,
                                         world_->keyframes_[t-2].pose);

  /*
  bool is_keyframe = false;
  const Keyframe& last_frm = world_->keyframes_[1];
  const Keyframe& cur_frm = world_->keyframes_[0];
  const double rotation_thr = FLAGS_keyframe_rotation_thr / 180 * M_PI;
  Vec6 rel_pose = VisualOdometer::RelativePoseVector(last_frm.pose, cur_frm.pose);
  is_keyframe = rel_pose.tail(3).norm() > compare_pose.tail(3).norm()
                * FLAGS_keyframe_translation_thr;
  if (!is_keyframe)
    return false;
    */

    //Count Matched ftids
  int cnt = 0;
  for (int i = 0; i < ftids1.size(); ++i) {
    const int ftid = ftids1[i];
    int idx = -1;
    for (int j = 0; j < ftids2.size(); ++j){
      if (ftid == ftids2[j])
        idx = j;
    }
    if (idx < 0)
      continue;
    cnt++;
  }
  double overlap_ratio = (double)cnt / ftids1.size();

  LOG(INFO) << overlap_ratio << " " << FLAGS_overlap_th << endl;
  if (overlap_ratio < FLAGS_overlap_th)
    return true;
  else
    return false;

//  if (cnt < 150)
//    return true;
}

}  // namespace rvslam

