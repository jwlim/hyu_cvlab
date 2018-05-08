#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdarg.h>
#include <stdio.h>

#include "feat_opencv.h"
#include "homography.h"
#include "five_point.h"
#include "rvslam_util.h"
#include "fastdetector.h"

#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;

DEFINE_int32(fast_th, 20, "FAST detect threshol");
DEFINE_int32(fast_window, 5, "FAST detect threshol");
DEFINE_double(matching_th, 0.5, "Matching ratio for new detection");
DEFINE_double(matching_radius, 10, "Matching ratio for new detection"); DEFINE_int32(good_matches_th, 0, "good matching th for process");

namespace rvslam {

typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
typedef Eigen::ArrayXXf ArrayF;

//define functions
void RadiusMatching(const cv::Mat& descriptors1,
                    const vector<cv::KeyPoint>& keypoints1,
                    const cv::Mat& descriptors2,
                    const vector<cv::KeyPoint>& keypoints2,
                    const double radius,
                    const double dist,
                    const Mat4 cur_pose,
                    vector<cv::DMatch>* matches); 

void FindGoodMatches(const vector<cv::KeyPoint>& keypoints1,
                    const vector<cv::KeyPoint>& keypoints2,
                    const vector<cv::DMatch>& matches,
                    vector<cv::DMatch>* good_matches); 


template <class T> inline
  string ToString(const T& v) {
    stringstream ss;
    ss << v;
    return ss.str();
  }

inline std::string StringPrintf(const char* fmt, ...) {
  char buf[1024];
  va_list a;
  va_start(a, fmt);
  vsprintf(buf, fmt, a);
  va_end(a);
  return std::string(buf);
}

inline double Distance(double x0, double y0, double x1, double y1) {
  const double dx = x0 - x1, dy = y0 - y1;
  return sqrt(dx * dx + dy * dy);
}

inline double Distance(const Vec3& pt1, const Vec3& pt2) {
  return Distance(pt1(0), pt1(1), pt2(0), pt2(1));
}

bool DetectFastBrief(const Eigen::MatrixXd& img_eigen,
                     vector<cv::KeyPoint>* keypoints,
                     cv::Mat* descriptors) {
    keypoints->clear();

		cv::Mat img;
		eigen2cv(img_eigen, img);
    img.convertTo(img, CV_8UC1);

{
    int block = FLAGS_fast_window;
    int count;
    DetectFastFeature(img.data, img.cols, img.rows, img.cols,
                      FLAGS_fast_th, block, keypoints); 


    // Extract descriptors.
    cv::BriefDescriptorExtractor extractor;
    extractor.compute(img, *keypoints, *descriptors);
}


    return true;
}

bool DetectSIFTFrame(const string image_file,
                    const int idx,
                    const int reduce_size,
                    vector<Keypoint>* keypoints,
                    MatF* descriptors) {
  // Descriptors are saved in a (number of features) x 128 matrix.
  keypoints->clear();

  cv::Mat img = cv::imread(StringPrintf(image_file.c_str(), idx), 0);
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
  for (int i = 0; i < keypoint.size(); ++i) {
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

bool DetectSURFFrame(const string image_file,
                     const int idx,
                     const int reduce_size,
                     vector<Keypoint>* keypoints,
                     MatF* descriptors) {
  // Descriptors are saved in a (number of features) x 128 matrix.
  keypoints->clear();

  cv::Mat img = cv::imread(StringPrintf(image_file.c_str(), idx), 0);
  if (reduce_size) {
    for (int i = 0; i < reduce_size; ++i) {
      cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
  }

  // Detect Features.
  cv::SiftFeatureDetector detector;
  vector<cv::KeyPoint> keypoint;
  detector.detect(img, keypoint);

  // Extract descriptor.
  cv::SurfDescriptorExtractor extractor;
  cv::Mat desc;
  extractor.compute(img, keypoint, desc);

  // Keypoints show (For Test).
  cv::Mat output;
  cv::drawKeypoints(img, keypoint, output, cv::Scalar::all(-1),
      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // Convert to Eigen.
  for (int i = 0; i < keypoint.size(); ++i) {
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


bool DetectOrb(const string image_file,
               const int idx,
               const int reduce_size,
               vector<cv::KeyPoint>* keypoints,
               cv::Mat* descriptors) {

  cv::Mat img = cv::imread(StringPrintf(image_file.c_str(), idx), 0);
  if (reduce_size) {
    for (int i = 0; i < reduce_size; ++i) {
      cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
    }
  }

  // Detect Features.
  cv::OrbFeatureDetector detector(FLAGS_fast_th);
  detector.detect(img, *keypoints);

  // Extract descriptors.
  cv::OrbDescriptorExtractor extractor;
  extractor.compute(img, *keypoints, *descriptors);

  return true;
}


void FeatureDetector::Setup(const Eigen::MatrixXd& img_eigen) {
	DetectFastBrief(img_eigen, &keypoints_, &descriptor_);
  const int n = keypoints_.size();

  cur_ftids_.reserve(n);
  for (int i = 0; i < n; ++i) {
    cur_ftids_.push_back(i);
  }
  ftids_ = cur_ftids_;
  last_ftid_ = cur_ftids_[n - 1];
  total_ftid_ = n;

  key_descriptor_ = descriptor_;
  key_ftids_= ftids_;
  key_keypoints_ = keypoints_;
  id_ = 0;
  
  new_ftid_ = false;
}

bool FeatureDetector::Process(const Eigen::MatrixXd& img_eigen,
                              const Mat4 cur_pose) {


  im1 = im2;
  cv::eigen2cv(img_eigen, im2);
  im2.convertTo(im2, CV_8UC1);

  cv::Mat cur_descriptor;
  vector<cv::KeyPoint> cur_keypoints;

  ProfileBegin("11.Detectfeat", &pdb_);
	DetectFastBrief(img_eigen, &cur_keypoints, &cur_descriptor);

	//LOG(INFO) << "Features: " << cur_keypoints.size();
  ProfileEnd("11.Detectfeat", &pdb_);

  if (descriptor_.type() != CV_8U) {
    descriptor_.convertTo(descriptor_, CV_8U);
  }
  if (cur_descriptor.type() != CV_8U) {
    cur_descriptor.convertTo(cur_descriptor, CV_8U);
  }

  vector<cv::DMatch> good_matches;
  ProfileBegin("12.Matching", &pdb_);
  RadiusMatching(descriptor_, keypoints_, cur_descriptor, cur_keypoints,
      FLAGS_matching_radius, 30, cur_pose, &good_matches);

  ProfileEnd("12.Matching", &pdb_);
	//LOG(INFO) << "[" << idx << "] "
  //         << "good matches: " << good_matches.size();

  ProfileBegin("13.SetFtids", &pdb_);
  //Set ftids
  vector<int> cur_ftids;
  int n = cur_keypoints.size();
  cur_ftids.reserve(n);
  cv::Mat new_descriptor;
  if (new_ftid_) {
    int cur_ftid = last_ftid_ + 1; 
    for (int i = 0; i < cur_keypoints.size(); ++i) {
      int ftid = cur_ftid;
      cur_ftid++;
      for (int j = 0; j < good_matches.size(); j++) {
        const int q = good_matches[j].queryIdx;
        const int t = good_matches[j].trainIdx;
        if (q == i) {
          ftid = ftids_[t]; 
          cur_ftid--;
          break;
        }
        //new ftids, keypoints are defined(except discriptor)
      }
      cur_ftids.push_back(ftid);
    }
    total_ftid_ = cur_ftids.size();
    last_ftid_ = cur_ftid;
    descriptor_ = cur_descriptor;
    keypoints_ = cur_keypoints;
  } else {
    //set new keypoints
    keypoints_.clear();
    keypoints_.reserve(n);
    for (int j = 0; j < good_matches.size(); ++j) {
      const int q = good_matches[j].queryIdx;
      const int t = good_matches[j].trainIdx;
      const int ftid = ftids_[t]; 
      //new ftids, keypoints are defined(except discriptor)
      cur_ftids.push_back(ftid);
      keypoints_.push_back(cur_keypoints[q]);
      new_descriptor.push_back(cur_descriptor.row(q));
    }
    descriptor_ = new_descriptor;
  }
  new_ftid_ = false;
  if ((double)cur_ftids.size() < FLAGS_matching_th)
    new_ftid_ = true;
  ProfileEnd("13.SetFtids", &pdb_);


  ftids_ = cur_ftids;
  return true;
}

bool FeatureDetector::Process(const Eigen::MatrixXd& img_eigen) {
  cv::Mat cur_descriptor;
  vector<cv::KeyPoint> cur_keypoints;

  ProfileBegin("11.Detectfeat", &pdb_);
	DetectFastBrief(img_eigen, &cur_keypoints, &cur_descriptor);

	//LOG(INFO) << "Features: " << cur_keypoints.size();
  ProfileEnd("11.Detectfeat", &pdb_);

  if (descriptor_.type() != CV_8U) {
    descriptor_.convertTo(descriptor_, CV_8U);
  }
  if (cur_descriptor.type() != CV_8U) {
    cur_descriptor.convertTo(cur_descriptor, CV_8U);
  }

  vector<cv::DMatch> good_matches;
  ProfileBegin("12.Matching", &pdb_);
  RadiusMatching(descriptor_, keypoints_, cur_descriptor, cur_keypoints,
      FLAGS_matching_radius, 30, &good_matches);
  ProfileEnd("12.Matching", &pdb_);
	//cout << "[" << idx << "] "
   //        << "good matches: " << good_matches.size() <<"/"
    //       << cur_keypoints.size() << endl;

  ProfileBegin("13.SetFtids", &pdb_);
  //Set ftids
  vector<int> cur_ftids;
  int n = cur_keypoints.size();
  cur_ftids.reserve(n);
  cv::Mat new_descriptor;
  if (new_ftid_) {
    int cur_ftid = last_ftid_ + 1; 
    for (int i = 0; i < cur_keypoints.size(); ++i) {
      int ftid = cur_ftid;
      cur_ftid++;
      for (int j = 0; j < good_matches.size(); j++) {
        const int q = good_matches[j].queryIdx;
        const int t = good_matches[j].trainIdx;
        if (q == i) {
          ftid = ftids_[t]; 
          cur_ftid--;
          break;
        }
        //new ftids, keypoints are defined(except discriptor)
      }
      cur_ftids.push_back(ftid);
    }
    total_ftid_ = cur_ftids.size();
    last_ftid_ = cur_ftid;
    descriptor_ = cur_descriptor;
    keypoints_ = cur_keypoints;
  } else {
    //set new keypoints
    keypoints_.clear();
    keypoints_.reserve(n);
    for (int j = 0; j < good_matches.size(); ++j) {
      const int q = good_matches[j].queryIdx;
      const int t = good_matches[j].trainIdx;
      const int ftid = ftids_[t]; 
      //new ftids, keypoints are defined(except discriptor)
      cur_ftids.push_back(ftid);
      keypoints_.push_back(cur_keypoints[q]);
      new_descriptor.push_back(cur_descriptor.row(q));
    }
    descriptor_ = new_descriptor;
  }
  new_ftid_ = false;
  if ((double)cur_ftids.size() < FLAGS_matching_th)
    new_ftid_ = true;
  ProfileEnd("13.SetFtids", &pdb_);

  ftids_ = cur_ftids;
  id_++;
  double matching_ratio = (double)good_matches.size() / key_keypoints_.size();
  
  return true;
}

//FindGoodMatches using distance
void FindGoodMatches(const vector<cv::KeyPoint>& keypoints1,
                     const vector<cv::KeyPoint>& keypoints2,
                     const vector<cv::DMatch>& matches,
                     vector<cv::DMatch>* good_matches) {
  //using distance
  const double dist_thr = 10; 
  good_matches->clear();
  for (int j = 0; j < matches.size(); ++j) {
    //query = 1, train = 2;
    cv::Point q = keypoints1[matches[j].queryIdx].pt;
    cv::Point t = keypoints1[matches[j].trainIdx].pt;
    const double dist = Distance(q.x, q.y, t.x, t.y);
    if (dist < dist_thr) good_matches->push_back(matches[j]);
  }

}

//Bit set count operation from
//http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int DescriptorDistance(const cv::Mat& a, const cv::Mat& b) {
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();
  int dist = 0;
  for (int i = 0; i < 8; i++, pa++, pb++) {
    unsigned int v = (*pa) ^ (*pb);
    v = v - ((v >> 1) & 0x55555555 );
    v = (v & 0x33333333) + ((v >> 2 ) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }
  return dist;
}

int DescriptorDistanceOpencv(const cv::Mat& a, const cv::Mat& b) {
  return cv::norm(a, b, cv::NORM_HAMMING);
}

void RadiusMatching(const cv::Mat& descriptors1,
                    const vector<cv::KeyPoint>& keypoints1,
                    const cv::Mat& descriptors2,
                    const vector<cv::KeyPoint>& keypoints2,
                    const double radius,
                    const double dist,
                    const Mat4 cur_pose,
                    vector<cv::DMatch>* matches) {
  const int n = descriptors1.rows;
  const int m = descriptors2.rows;
  vector<cv::DMatch> new_matches;

  for (int i = 0; i < n; ++i) {
    const cv::Mat& des1 = descriptors1.row(i);
    cv::Point p1 = keypoints1[i].pt;
    Vec3 p;
    p << p1.x, p1.y, 1;
    p = cur_pose.block(0,0,3,4) * Hom(p);

    int min_dist = dist;
    int min_idx = -1;
    for (int j = 0; j < m; ++j) {
      const cv::Mat& des2 = descriptors2.row(j);
      cv::Point p2 = keypoints2[j].pt;
      const double dist = Distance(p(0), p(1), p2.x, p2.y);
      if (dist > radius) continue;
      const double diff = DescriptorDistance(des1, des2);
      if (diff < min_dist) {
        min_dist = dist;
        min_idx = j;
      }
    }
    if (min_idx < 0) continue;
    // check repetition
    int prev_idx = -1;
    for (int k = 0; k < new_matches.size() && prev_idx < 0; ++k) {
      if (new_matches[k].queryIdx == min_idx) {
        prev_idx = k;
      }
    }
    if (prev_idx >= 0) {
      new_matches[prev_idx].queryIdx = min_idx;
      new_matches[prev_idx].trainIdx = i;
      new_matches[prev_idx].distance = min_dist;
    } else {
      cv::DMatch match;
      match.queryIdx = min_idx;
      match.trainIdx = i;
      match.distance = min_dist;
      new_matches.push_back(match);
    }

  }
  *matches = new_matches;
}

void FeatureDetector::RadiusMatchingTest (const cv::Mat& descriptors1,
                    const vector<cv::KeyPoint>& keypoints1,
                    const cv::Mat& descriptors2,
                    const vector<cv::KeyPoint>& keypoints2,
                    const double radius,
                    const double dist,
                    const Mat4 cur_pose,
                    vector<cv::DMatch>* matches) {
  const int n = descriptors1.rows;
  const int m = descriptors2.rows;
  vector<cv::DMatch> new_matches;

//test
  vector<cv::KeyPoint> test_keypoint;

  for (int i = 0; i < n; ++i) {
    const cv::Mat& des1 = descriptors1.row(i);
    cv::Point p1 = keypoints1[i].pt;
    Vec3 p;
    p << p1.x, p1.y, 1;
    p = cur_pose.block(0,0,3,4) * Hom(p);
    

//test
    cv::KeyPoint kp;
    kp = keypoints1[i];
    kp.pt.x = p(0), kp.pt.y = p(1);
    test_keypoint.push_back(kp);


    int min_dist = dist;
    int min_idx = -1;
    for (int j = 0; j < m; ++j) {
      const cv::Mat& des2 = descriptors2.row(j);
      cv::Point p2 = keypoints2[j].pt;
      const double dist = Distance(p(0), p(1), p2.x, p2.y);
      if (dist > radius) continue;
      const double diff = DescriptorDistance(des1, des2);
      if (diff < min_dist) {
        min_dist = dist;
        min_idx = j;
      }
    }
    if (min_idx < 0) continue;
    // check repetition
    int prev_idx = -1;
    for (int k = 0; k < new_matches.size() && prev_idx < 0; ++k) {
      if (new_matches[k].queryIdx == min_idx) {
        prev_idx = k;
      }
    }
    if (prev_idx >= 0) {
      new_matches[prev_idx].queryIdx = min_idx;
      new_matches[prev_idx].trainIdx = i;
      new_matches[prev_idx].distance = min_dist;
    } else {
      cv::DMatch match;
      match.queryIdx = min_idx;
      match.trainIdx = i;
      match.distance = min_dist;
      new_matches.push_back(match);
    }

  }
  *matches = new_matches;

//test
//  cv::Mat output, output2;
// cv::drawKeypoints(im2, test_keypoint, output, cv::Scalar(255,0,0));
//string fn = "../test.png";
//  cv::imwrite(fn, output);
}

void RadiusMatching(const cv::Mat& descriptors1,
                    const vector<cv::KeyPoint>& keypoints1,
                    const cv::Mat& descriptors2,
                    const vector<cv::KeyPoint>& keypoints2,
                    const double radius,
                    const double dist,
                    vector<cv::DMatch>* matches) {
  const int n = descriptors1.rows;
  const int m = descriptors2.rows;
  vector<cv::DMatch> new_matches;

  for (int i = 0; i < n; ++i) {
    const cv::Mat& des1 = descriptors1.row(i);
    cv::Point p1 = keypoints1[i].pt;

    int min_dist = dist;
    int min_idx = -1;
    for (int j = 0; j < m; ++j) {
      const cv::Mat& des2 = descriptors2.row(j);
      cv::Point p2 = keypoints2[j].pt;
      const double dist = Distance(p1.x, p1.y, p2.x, p2.y);
      if (dist > radius) continue;
      //just compare in radius
      //find best matching(q)
      const double diff = DescriptorDistance(des1, des2);
      if (diff < min_dist) {
        min_dist = dist;
        min_idx = j;
      }
    }
    if (min_idx < 0) continue;
    // check repetition
    int prev_idx = -1;
    for (int k = 0; k < new_matches.size() && prev_idx < 0; ++k) {
      if (new_matches[k].queryIdx == min_idx) {
        prev_idx = k;
      }
    }
    if (prev_idx >= 0) {
      new_matches[prev_idx].queryIdx = min_idx;
      new_matches[prev_idx].trainIdx = i;
      new_matches[prev_idx].distance = min_dist;
    } else {
      cv::DMatch match;
      match.queryIdx = min_idx;
      match.trainIdx = i;
      match.distance = min_dist;
      new_matches.push_back(match);
    }

  }
  *matches = new_matches;
}

cv::Mat FeatureDetector::ComputeDescriptor(const Eigen::MatrixXd& img_eigen,
                                           vector<cv::KeyPoint>& keypoints) {
  cv::Mat img;
  eigen2cv(img_eigen, img);
  img.convertTo(img, CV_8UC1);

  cv::Mat descriptors;
  cv::BriefDescriptorExtractor extractor;
  extractor.compute(img, keypoints, descriptors);

  return descriptors;
}

} // namespace rvslam

