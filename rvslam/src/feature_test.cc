#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "image_file.h"
#include "image_util.h"
#include "klt_tracker.h"
#include "rvslam_util.h"
#include "visual_odometer.h"
#include "feature.h"

using namespace std;
using namespace rvslam;

DEFINE_string(image_files, "../data/test/%04d.png", "Input images.");
DEFINE_int32(start, 0, "Start index of files to process.");
DEFINE_int32(end, 2, "End index of files to process.");
DEFINE_int32(reduce_size, 0, "reduce size 1/n");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  vector<vector<Keypoint> > keypoints;
  vector<MatF> descriptors;
  FeatureDetector feat;  


  //TEST DetectFrame
  //SIFT
  MatF descriptor;
  MatF descriptor2;
  vector<Keypoint> keypoint;
  vector<Keypoint> keypoint2;
  feat.DetectSIFTFrame(FLAGS_image_files, 0,
                       FLAGS_reduce_size,
                       &keypoint, &descriptor);
  feat.DetectSIFTFrame(FLAGS_image_files, 1,
                       FLAGS_reduce_size,
                       &keypoint2, &descriptor2);
  keypoints.push_back(keypoint);
  keypoints.push_back(keypoint2);
  descriptors.push_back(descriptor);
  descriptors.push_back(descriptor2);
  feat.MatchingTest(FLAGS_image_files, FLAGS_reduce_size, keypoints,
                    descriptors, 0, 1, "SIFT_FRAME");

  //SURF
  feat.DetectSURFFrame(FLAGS_image_files, 0,
                       FLAGS_reduce_size,
                       &keypoint, &descriptor);
  feat.DetectSURFFrame(FLAGS_image_files, 1,
                       FLAGS_reduce_size,
                       &keypoint2, &descriptor2);
  keypoints.clear();
  descriptors.clear();
  keypoints.push_back(keypoint);
  keypoints.push_back(keypoint2);
  descriptors.push_back(descriptor);
  descriptors.push_back(descriptor2);
  feat.MatchingTest(FLAGS_image_files, FLAGS_reduce_size, keypoints,
                    descriptors, 0, 1, "SURF_FRAME");

  //TEST Detect features whole frames 
  //SIFT
  feat.DetectSIFT(FLAGS_image_files, FLAGS_start, FLAGS_end,
                  FLAGS_reduce_size,
                  &keypoints, &descriptors);
  feat.MatchingTest(FLAGS_image_files, FLAGS_reduce_size, keypoints,
                    descriptors, 0, 1, "SIFT");

  //SURF
  feat.DetectSURF(FLAGS_image_files, FLAGS_start, FLAGS_end,
                  FLAGS_reduce_size,
                  &keypoints, &descriptors);
  feat.MatchingTest(FLAGS_image_files, FLAGS_reduce_size, keypoints,
                    descriptors, 0, 1, "SURF");
  

  cv::waitKey(0);

  return 0;
}

