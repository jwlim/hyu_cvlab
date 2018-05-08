// feature_harris_klt.h
//
// Author : Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_FEAT_HARRIS_KLT_H_
#define _RVSLAM_FEAT_HARRIS_KLT_H_

#include <vector>
#include <Eigen/Dense>

#include "feature.h"  // Feature extractor interface

#include "rvslam_common.h"
#include "image_pyramid.h"

namespace rvslam {

//-----------------------------------------------------------------------------
// Harris / KLT feature detector and tracker (no descriptor).

class HarrisKLTContext : public BaseFeatureExtractorContext<Mat2Xf, Matf> {
 public:
  HarrisKLTContext() {}

  int id, level;
  Eigen::Vector2f pos;
  double score;

  HarrisCorner() : id(-1), level(0), pos(), score(0.0) {}
  HarrisCorner(const HarrisCorner& ft)
      : id(ft.id), level(ft.level), pos(ft.pos), score(ft.score) {}
  HarrisCorner(int id, int level, const Eigen::Vector2f& pos, double score)
      : id(id), level(level), pos(pos), score(score) {}

  void Set(int _id, int _level, const Eigen::Vector2f& _pos, double _score) {
    id = _id, level = _level, pos = _pos, score = _score;
  }
 protected:
  // Additional member variables;
  ImagePyramid pyr_;
  ImagePyramidBuilder pyramid_builder_;
};

class HarrisCornerDetector : public FeatureExtractor<ArrayXXu8, Mat2Xf, Matf> {
 public:

  HarrisCornerDetector();
  virtual ~HarrisCornerDetector();
  virtual bool SetImage(const ImageType& image);
  virtual bool Detect(FeatLocMatType* ftlocs);
  virtual bool GetDescriptors(const FeatLocMatType& ftlocs,
                              FeatDescMatType* ftdescs);
  virtual void Cleanup();

  const ImagePyramid& pyramid() const { return pyr_; }

 protected:
  ImagePyramid pyr_;
  ImagePyramidBuilder pyramid_builder_;
};

//-----------------------------------------------------------------------------
// KLT feature tracker.

class KLTTracker : public FeatureTracker<ArrayXXu8, HarrisCornerDetector> {
 public:
  KLTTracker();
  virtual ~KLTTracker();
  virtual bool Process(const ImageType& image);
};

} //namespace rvslam

#endif  // _RVSLAM_FEAT_HARRIS_KLT_H_

