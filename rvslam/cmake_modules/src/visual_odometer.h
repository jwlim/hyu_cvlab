// visual_odometer.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _RVSLAM_VISUAL_ODOMETER_H_
#define _RVSLAM_VISUAL_ODOMETER_H_

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
//#include "loopclosing.h"

namespace rvslam {

// VisualOdometer

class VisualOdometer {
 public:
  struct Calib {
    double fx, fy, cx, cy, k1, k2;
  };

  VisualOdometer(const Calib& calib,
                 Map* world,
                 const char* voc_file = NULL);

  // ProcessFrame function takes the tracked features (with disparity).
  // Each column of image_pts has (pixel_x_coord, pixel_y_coord, stereo_disp).
  // The normalized coordinate and depth is computed internally.
  bool ProcessFrame(int frmno,
                    const std::vector<int>& ftids, const Mat3X& image_pts,
                    cv::Mat& descriptors);

  //do not use voctree
  bool ProcessFrame(int frmno,
                    const std::vector<int>& ftids, const Mat3X& image_pts);

  Vec3 EstimateRotationalVelocity() const;

  void GetFeatureProjections(const std::vector<int>& ftids, const Vec6& pose,
                             Mat2X* pos) const;
  Mat2X veh_fpos;
  Mat2X p3p_fpos;

  //const Vec6& pose() const { return keyframes_[0].pose; }
  //const std::vector<int>& pose_inliers() const { return pose_inliers_; }
  //const std::vector<Keyframe>& keyframes() const { return keyframes_; }
  //const std::map<int, Vec3>& ftid_pts_map() const { return ftid_pts_map_; }

  // Static utility functions.
  static Vec6 ToPoseVector(const Mat34& pose_mat);
  static Mat34 ToPoseMatrix(const Vec6& pose_vec);
  static Mat34 InversePoseMatrix(const Mat34& pose_mat);
  static Vec6 RelativePoseVector(const Vec6& ref_pose, const Vec6& pose);

 private:
  void MakeStereoPoints(const Mat3X& image_pts, Mat4X* stereo_pts) const;
  void MakeNormalizedPoints(const Mat3X& image_pts,
                            Mat3X* normalized_pts) const; 
  void GetUndistortedPoints(const Mat3X& pts, Mat3X* pts_out) const;
  void FindMatchedPoints(const Mat3X& pts1, const std::vector<int>& ftids1,
                         const Mat3X& pts2, const std::vector<int>& ftids2,
                         Mat3X* new_pts1, Mat3X* new_pts2,
                         std::vector<int>* new_ftids) const; 
  bool InitializeWorld(const Mat3X& pts1, const Mat3X& pts2,
                       const std::vector<int>& ftids);
  bool CheckKeyframe(const std::vector<int>& ftids1,
                     const std::vector<int>& ftids2) const;
  void UpdateConnections(Keyframe* cur_frm, int num_keyframes);

  void BuildWorldPointsMatrixForP3P(const std::vector<int>& ftids,
                                    std::vector<int>* ftid_idx_found,
                                    Mat3X* world_pts) const;
  bool PoseOptimize(); 
  void UpdateWorldPoints();

  void FindPoseInliers(const std::vector<int>& ftids, const Mat3X& normalized_pts, 
                       const Mat3X& world_pts, const Vec6& pose,
                       const double inlier_thr,
                       std::vector<int>* inliers) const;

  // The first instance is the current frame (not a keyframe), and
  // the second one is the most recent keyframe.
  std::vector<Keyframe> keyframes_all_;
  Calib calib_;
  double pose_inlier_thr_;
  std::vector<int> pose_inliers_;
  double scale;

  Map* world_;

  //LoopCloser
  //LoopClosing loop;
};

}  // namespace rvslam
#endif  // _RVSLAM_VISUAL_ODOMETER_H_

