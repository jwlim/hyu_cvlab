// feat_opencv.h
// Author : Euntae Hong (dragon1301@naver.com)
//

#include <vector>
#include <string>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "rvslam_common.h"

using namespace std;

typedef Eigen::MatrixXf MatF;

namespace rvslam {

struct Keypoint {
  Vec2 loc;
  double sc;
  double angle;
  int ftid;
};

void RadiusMatching(const cv::Mat& descriptors1,
                    const vector<cv::KeyPoint>& keypoints1,
                    const cv::Mat& descriptors2,
                    const vector<cv::KeyPoint>& keypoints2,
                    const double radius,
                    const double dist,
                    vector<cv::DMatch>* matches);


// 2 version of Detect features
// DetectXXXX : Detect features whole frames(Output is vector)
// DetectXXXXFrame : Detect features on 1 frame

bool DetectSIFTFrame(const string image_file,
                     const int idx,
                     const int reduce_size,
                     vector<Keypoint>* keypoints,
                     MatF* descriptors);

bool DetectSURFFrame(const string image_file,
                     const int idx,
                     const int reduce_size,
                     vector<Keypoint>* keypoints,
                     MatF* descriptors);

bool DetectSIFT(const string image_file,
                const int start, const int end,
                const int reduce_size,
                vector<vector<Keypoint> >* keypoints,
                vector<MatF>* descriptors);

bool DetectSURF(const string image_file,
                const int start, const int end,
                const int reduce_size,
                vector<vector<Keypoint> >* keypoints,
                vector<MatF>* descriptors);

class FeatureDetector {
 public:
  void Setup(const Eigen::MatrixXd& img_eigen);
  bool Process(const Eigen::MatrixXd& img_eigen);

  bool Process(const Eigen::MatrixXd& img_eigen,
               const Mat4 cur_pose);

  cv::Mat ComputeDescriptor(const Eigen::MatrixXd& img_eigen, vector<cv::KeyPoint>& keypoints);

  const vector<cv::KeyPoint>& keypoints() const { return keypoints_; }; 
  const cv::Mat& descriptor() const { return descriptor_; };
  const vector<int>& ftids() const { return ftids_; };
  int last_ftid() const { return last_ftid_; };
  int latest_knum() const { return latest_knum_; };

 private:
  cv::Mat descriptor_;
  vector<cv::KeyPoint> keypoints_;
  vector<int> ftids_;
  //keyframe
  cv::Mat key_descriptor_;
  vector<cv::KeyPoint> key_keypoints_;
  vector<int> key_ftids_;

  vector<int> cur_ftids_;
  int last_ftid_;
  int total_ftid_;
  bool new_ftid_;
  int latest_knum_;

  int id_;

//test
  cv::Mat im1, im2;
void RadiusMatchingTest (const cv::Mat& descriptors1,
                    const vector<cv::KeyPoint>& keypoints1,
                    const cv::Mat& descriptors2,
                    const vector<cv::KeyPoint>& keypoints2,
                    const double radius,
                    const double dist,
                    const Mat4 cur_pose,
                    vector<cv::DMatch>* matches); 
};

} //namespace rvslam
