// feature.cc
//
// Author : Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include "feature.h"

using namespace std;
namespace rvslam {

//template <typename ImageType, typename ContextType>
//bool DetectFeatures(const ImageType& image, ContextType* context) {
//  return false;
//}

struct HarrisKLTContext {
  typedef Mat2Xf FeatLocMatType;  // (x, y)
  typedef Eigen::Matrix<double, 0, Eigen::Dynamic> FeatDescMatType;  // N.A.
};

template <>
bool DetectFeatures(const ArrayXXu8& image, HarrisKLTContext* context) {
  return false;
}


template <>
FeatureExtractor<ArrayXXu8, HarrisKLTContext>::FeatureExtractor() {
}

template <>
FeatureExtractor<ArrayXXu8, HarrisKLTContext>::~FeatureExtractor() {
}

template <>
bool FeatureExtractor<ArrayXXu8, HarrisKLTContext>::SetImage(
    const ArrayXXu8& image) {
  return false;
}

template <>
bool FeatureExtractor<ArrayXXu8, HarrisKLTContext>::Track(
    const ArrayXXu8& image) {
  return false;
}

void TestFunc() {
  ArrayXXu8 tmp_image;
  HarrisKLTContext feat_context;
  FeatureExtractor<ArrayXXu8, HarrisKLTContext> test;
  test.SetImage(tmp_image);
  test.Track(tmp_image);

//  DetectFeatures(tmp_image, &feat_context);
//  DetectFeatures(tmp_image, NULL);
}

/*
HarrisCornerDetector::HarrisCornerDetector() {
}

HarrisCornerDetector::~HarrisCornerDetector() {
}

bool HarrisCornerDetector::SetImage(const ImageType& image) {
  return false;
}

bool HarrisCornerDetector::Detect(FeatLocMatType* ftlocs) {
  return false;
}

bool HarrisCornerDetector::GetDescriptors(const FeatLocMatType& ftlocs,
                                          FeatDescMatType* ftdescs) {
  return false;
}

void HarrisCornerDetector::Cleanup() {
}

KLTTracker::KLTTracker() {
}

KLTTracker::~KLTTracker() {
}

bool KLTTracker::Process(const ImageType& image) {
  return false;
}
*/

}  // namespace rvslam

