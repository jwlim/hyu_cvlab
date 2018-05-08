#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdarg.h>
#include <stdio.h>
#include "feature.h"

#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;

namespace rvslam {

typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
typedef Eigen::ArrayXXf ArrayF;

template <class T> inline
string ToString(const T& v) {
  stringstream ss;
  ss << v;
  return ss.str();
}

inline std::string StringPrintf(const std::string& fmt, ...) {
  char buf[1024];
  va_list a;
  va_start(a, fmt);
  vsprintf(buf, fmt.c_str(), a);
  va_end(a);
  return std::string(buf);
}

bool FeatureDetector::DetectSIFTFrame(const string image_file,
                                      const int idx,
                                      const int reduce_size,
                                      vector<Keypoint>* keypoints,
                                      MatF* descriptors){

  //descriptor is saved as (number of features) x 128 
  keypoints->clear();
  
  cv::Mat img = cv::imread(StringPrintf(image_file, idx), 0);
  if (reduce_size) {
    for (int i = 0; i < reduce_size; ++i) {
      cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
  }

  //Detect Features
  cv::SiftFeatureDetector detector;
  vector<cv::KeyPoint> keypoint;
  detector.detect(img, keypoint);

  //Extract descriptor
  cv::SiftDescriptorExtractor extractor;
  cv::Mat desc;
  extractor.compute(img, keypoint, desc);

  //Keypoints show(For Test)
  cv::Mat output;
  cv::drawKeypoints(img, keypoint, output, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  //Convert to Eigen
  const int n = keypoint.size();
  for (int i = 0; i < n; ++i) {
    Keypoint kp;
    kp.loc(0) = keypoint[i].pt.x;
    kp.loc(1) = keypoint[i].pt.y;
    kp.sc = keypoint[i].size;
    kp.angle = keypoint[i].angle;
    keypoints->push_back(kp);
  }
  cv2eigen(desc, *descriptors);

  return true;
}

bool FeatureDetector::DetectSURFFrame(const string image_file,
                                      const int idx,
                                      const int reduce_size,
                                      vector<Keypoint>* keypoints,
                                      MatF* descriptors){

  //descriptor is saved as (number of features) x 128 
  keypoints->clear();
  
  cv::Mat img = cv::imread(StringPrintf(image_file, idx), 0);
  if (reduce_size) {
    for (int i = 0; i < reduce_size; ++i) {
      cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
  }

  //Detect Features
  cv::SiftFeatureDetector detector;
  vector<cv::KeyPoint> keypoint;
  detector.detect(img, keypoint);

  //Extract descriptor
  cv::SurfDescriptorExtractor extractor;
  cv::Mat desc;
  extractor.compute(img, keypoint, desc);

  //Keypoints show(For Test)
  cv::Mat output;
  cv::drawKeypoints(img, keypoint, output, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  //Convert to Eigen
  const int n = keypoint.size();
  for (int i = 0; i < n; ++i) {
    Keypoint kp;
    kp.loc(0) = keypoint[i].pt.x;
    kp.loc(1) = keypoint[i].pt.y;
    kp.sc = keypoint[i].size;
    kp.angle = keypoint[i].angle;
    keypoints->push_back(kp);
  }
  cv2eigen(desc, *descriptors);

  return true;
}


bool FeatureDetector::DetectSIFT(const string image_file,
                                 const int start, const int end,
                                 const int reduce_size,
                                 vector<vector<Keypoint> >* keypoints,
                                 vector<MatF>* descriptors) {

  //descriptor is saved as (number of features) x 128 x m(number of frames)
  keypoints->clear();
  descriptors->clear();
  
  for (int idx = start; idx < end; idx++) {
    cv::Mat img = cv::imread(StringPrintf(image_file, idx), 0);
    if (reduce_size) {
      for (int i = 0; i < reduce_size; ++i) {
        cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
      }
    }

    //Detect Features
    cv::SiftFeatureDetector detector;
    vector<cv::KeyPoint> keypoint;
    detector.detect(img, keypoint);

    //Extract descriptor
    cv::SiftDescriptorExtractor extractor;
    cv::Mat desc;
    extractor.compute(img, keypoint, desc);

    //Keypoints show(For Test)
    cv::Mat output;
    cv::drawKeypoints(img, keypoint, output, cv::Scalar::all(-1),
                       cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //Convert to Eigen
    const int n = keypoint.size();
    vector<Keypoint> keypoint_eigen;
    MatF desc_eigen;
    for (int i = 0; i < n; ++i) {
      Keypoint kp;
      kp.loc(0) = keypoint[i].pt.x;
      kp.loc(1) = keypoint[i].pt.y;
      kp.sc = keypoint[i].size;
      kp.angle = keypoint[i].angle;
      keypoint_eigen.push_back(kp);
    }
      cv2eigen(desc, desc_eigen);

      keypoints->push_back(keypoint_eigen);
      descriptors->push_back(desc_eigen);
  }

  return true;
}

bool FeatureDetector::DetectSURF(const string image_file,
                                 const int start, const int end,
                                 const int reduce_size,
                                 vector<vector<Keypoint> >* keypoints,
                                 vector<MatF>* descriptors) {

  //descriptor is saved as (number of features) x 64 x m(number of frames)
  keypoints->clear();
  descriptors->clear();

  for (int idx = start; idx < end; idx++) {
    cv::Mat img = cv::imread(StringPrintf(image_file, idx), 0);

    if (reduce_size) {
      for (int i = 0; i < reduce_size; ++i) {
        cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
      }
    }

    //Detect Features
    cv::SiftFeatureDetector detector;
    vector<cv::KeyPoint> keypoint;
    detector.detect(img, keypoint);

    //Extract descriptor
    cv::SurfDescriptorExtractor extractor;
    cv::Mat desc;
    extractor.compute(img, keypoint, desc);

    //Keypoints show(For Test)
    cv::Mat output;
    cv::drawKeypoints(img, keypoint, output, cv::Scalar::all(-1),
                       cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //Convert to Eigen
    const int n = keypoint.size();
    vector<Keypoint> keypoint_eigen;
    MatF desc_eigen;
    for (int i = 0; i < n; ++i) {
      Keypoint kp;
      kp.loc(0) = keypoint[i].pt.x;
      kp.loc(1) = keypoint[i].pt.y;
      kp.sc = keypoint[i].size;
      kp.angle = keypoint[i].angle;
      keypoint_eigen.push_back(kp);
    }
      cv2eigen(desc, desc_eigen);

      keypoints->push_back(keypoint_eigen);
      descriptors->push_back(desc_eigen);
  }

  return true;
}

void FeatureDetector::MatchingTest(const string image_file,
                                   const int reduce_size,
                                   const vector<vector<Keypoint> >& keypoints,
                                   const vector<MatF>& descriptors,
                                   const int a, const int b,
                                   const string name) {

    cv::Mat img1 = cv::imread(StringPrintf(image_file, a), 0);
    cv::Mat img2 = cv::imread(StringPrintf(image_file, b), 0);

    if (reduce_size) {
      for (int i = 0; i < reduce_size; ++i) {
        cv::resize(img1, img1, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        cv::resize(img2, img2, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
      }
    }

    //Set Descriptors & Keypoints for matching
    cv::Mat descriptor1, descriptor2;
    vector<cv::KeyPoint> keypoints1, keypoints2;
    eigen2cv(descriptors[a], descriptor1);
    eigen2cv(descriptors[b], descriptor2);
    const vector<Keypoint>& keypoint1_eigen = keypoints[a];
    const vector<Keypoint>& keypoint2_eigen = keypoints[b];

    int n = keypoint1_eigen.size();
    for (int i = 0; i < n; ++i) {
      cv::KeyPoint kp;
      kp.pt.x = keypoint1_eigen[i].loc(0);
      kp.pt.y = keypoint1_eigen[i].loc(1);
      kp.size = keypoint1_eigen[i].sc;
      kp.angle = keypoint1_eigen[i].angle;
      keypoints1.push_back(kp);
    }

    n = keypoint2_eigen.size();
    for (int i = 0; i < n; ++i) {
      cv::KeyPoint kp;
      kp.pt.x = keypoint2_eigen[i].loc(0);
      kp.pt.y = keypoint2_eigen[i].loc(1);
      kp.size = keypoint2_eigen[i].sc;
      kp.angle = keypoint2_eigen[i].angle;
      keypoints2.push_back(kp);
    }


    //Matching
    cv::FlannBasedMatcher matcher;
    //cv::BFMatcher matcher(cv::NORM_L2);
    vector<cv::DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);

    cv::Mat output;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, output,
                     cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(),
                     cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); 

    imshow(name, output);
}


} // namespace rvslam













