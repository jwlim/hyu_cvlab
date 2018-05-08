// klt_tracker.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_KLT_TRACKER_H_
#define _RVSLAM_KLT_TRACKER_H_

#include <fstream>
#include <map>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Dense>
#include "image_pyramid.h"
#include "rvslam_common.h"

namespace rvslam {

struct KLTFeat {
  int id, level;
  Eigen::Vector2f pos;
  double score;
  bool badmotion;

  KLTFeat() : id(-1), level(0), pos(), score(0.0) {}
  KLTFeat(const KLTFeat& ft)
      : id(ft.id), level(ft.level), pos(ft.pos), score(ft.score) {}
  KLTFeat(int id, int level, const Eigen::Vector2f& pos, double score)
      : id(id), level(level), pos(pos), score(score){}

  void Set(int id_, int level_, const Eigen::Vector2f& pos_, double score_) {
    id = id_, level = level_, pos = pos_, score = score_;
  }
};

// KLTTracker - monocular KLT tracker.

class KLTTracker {
 public:
  typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
  typedef Eigen::Vector2f Vec2f;
  typedef Eigen::VectorXf VecXf;

  KLTTracker();
  bool Setup(const ArrayXXu8& image);
  bool Process(const ArrayXXu8& image);
  bool ProcessIMU(const ArrayXXu8& image, const Mat3 cur_rot);
  void Cleanup();

  bool IsSetup() const { return prev_.num_levels() > 0; }
  int frame_no() const { return frame_no_; }
  const std::vector<KLTFeat>& features() const { return features_; }
  void RemoveFeatures(const std::vector<int>& feature_idx);
  const ImagePyramid& image_pyramid() const { return prev_; }

 private:
  int frame_no_;
  int next_id_;
  int detect_count_;
  std::vector<KLTFeat> features_;
  ImagePyramid prev_;
  ImagePyramidBuilder pyramid_builder_;
};

// StereoKLTTracker - stereo KLT tracker.

class StereoKLTTracker {
 public:
  typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
  typedef std::pair<int, int> IDPair;

  StereoKLTTracker();
  bool Setup(const ArrayXXu8& left_image, const ArrayXXu8& right_array);
  bool Process(const ArrayXXu8& left_image, const ArrayXXu8& right_array);
  void Cleanup();

  bool IsSetup() const { return left_.IsSetup() && right_.IsSetup(); }
  int frame_no() const { return left_.frame_no(); }
  const KLTTracker& left() const { return left_; }
  const KLTTracker& right() const { return right_; }

  const std::vector<IDPair>& matched_features() const { return matches_; }

 private:
  static void MatchFeaturesTwoWay(
      const KLTTracker& left, const KLTTracker& right,
      std::vector<IDPair>* matches);
  static void MatchFeatures(const KLTTracker& ref, const KLTTracker& dst,
                            int match_dir, std::vector<IDPair>* matches);

  KLTTracker left_, right_;
  std::vector<IDPair> matches_;  // Matched feature indices.
};

}  // namespace rvslam
#endif  // _RVSLAM_KLT_TRACKER_H_
