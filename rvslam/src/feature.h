// feature.h
//
// Author : Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_FEATURE_H_
#define _RVSLAM_FEATURE_H_

#include <vector>
#include <Eigen/Dense>
#include "rvslam_common.h"

namespace rvslam {

// Generic feature extractor (detector, descriptor) and tracker interface.

template <typename FeatLocMatType_, typename FeatDescMatType_>
class BaseFeatureContext {
 public:
  typedef FeatLocMatType_ FeatLocMatType;
  typedef FeatDescMatType_ FeatDescMatType;

  const std::vector<int>& ftids() const { return ftids_; }
  const FeatLocMatType& ftloc() const { return ftloc_; }
  const FeatDescMatType& ftdesc() const { return ftdesc_; }

 protected:
  std::vector<int> ftids_;
  FeatLocMatType ftloc_;
  FeatDescMatType ftdesc_;
};

template <typename ImageType, typename ContextType> inline
bool DetectFeatures(const ImageType& image, ContextType* context) {
  return false;  // Must be overriden through template specialization.
}

template <typename ImageType, typename ContextType> inline
bool TrackFeatures(const ImageType& image, ContextType* context) {
  return false;  // Must be overriden through template specialization.
}

template <typename ImageType, typename ContextType> inline
bool GetFeatureDescriptors(const ImageType& image, ContextType* context) {
  return false;  // Must be overriden through template specialization.
}


/*
template <typename ImageType_, typename ContextType_>
class FeatureExtractor {
 public:
  typedef ImageType_ ImageType;
  typedef ContextType_ ContextType;
  typedef typename ContextType::FeatLocMatType FeatLocMatType;
  typedef typename ContextType::FeatDescMatType FeatDescMatType;

  FeatureExtractor() {}
  ~FeatureExtractor() { Cleanup(); }

  bool SetImage(const ImageType& image);
  bool Detect(FeatLocMatType* ftlocs);

  bool Track(const ImageType& image);

  bool GetDescriptors(const FeatLocMatType& ftlocs,
                      FeatDescMatType* ftdescs) const;

  void Cleanup() {}

  const FeatLocMatType& feature_locations() const { return ftlocs_; }
  const FeatDescMatType& feature_descriptors() const { return ftdescs_; }

 protected:
  std::vector<int> ftids_;
  FeatLocMatType ftlocs_;
  FeatDescMatType ftdescs_;
};

//void MatchFeatures(
//    const FeatLocMatType& prev_ftlocs, const FtDescMatType& prev_ftdescs,
//    const FeatLocMatType& cur_ftlocs, const FtDescMatType& cur_ftdescs,
//    std::vector<std::pair<int, int> >* matches);

/ *
template <typename ImageType, typename FeatureExtractor_>
class FeatureTracker : public FeatureExtractor_ {
 public:
  typedef FeatureExtractor_ FeatureExtractorType;

  FeatureTracker() {}
  virtual ~FeatureTracker() {}

  virtual bool Process(const ImageType& image) = 0;

 protected:
  std::vector<int> ftids_;
  FeatureExtractorType featex_;
};
*/

} //namespace rvslam

#endif  // _RVSLAM_FEATURE_H_

