#include "sim3solver.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime> 
#include <cmath>
#include <iostream>
#include <fstream>

using namespace std;
using namespace rvslam;

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  const int n = 300;

  //Set test Point Set1
  cv::Mat pt1 = cv::Mat(3, n, CV_32F);
  randu(pt1, cv::Scalar::all(0), cv::Scalar::all(255));

  //Set Sim3 Pose Matrix
  cv::Mat pose_mat;

  pose_mat = (cv::Mat_<float>(3,4) << 2, 0, 0, 20,
                                      0, 2, 0, -30,
                                      0, 0, 2, 110);

  //Set test Point Set1
  cv::Mat pt1_hom = pt1;
  cv::Mat row = cv::Mat::ones(1, n, CV_32F);
  pt1_hom.push_back(row);
  cv::Mat pt2 = cv::Mat(3, n, CV_32F);
  pt2 = pose_mat * pt1_hom;

  vector<cv::Mat> pt1_vec, pt2_vec;

  for (int i = 0; i < n; ++i) {
    cv::Mat p1(3,1,CV_32F), p2(3,1,CV_32F);
    p1 = pt1.col(i);
    p2 = pt2.col(i);
    pt1_vec.push_back(p1);
    pt2_vec.push_back(p2);
  }


  //Process
  cv::Mat calib(2,2,CV_32F);
  calib = (cv::Mat_<float>(2,2) << 300, 300, 320, 180);

  Sim3Solver sim;
  sim.Init(pt1_vec, pt2_vec, 0.99, 100, 300, calib);
  bool more;
  vector<bool> binliers;
  int inliers;
  cv::Mat pose_res = sim.EstimateSim3OpenCV(100, more, binliers, inliers);
  cout << pose_res.inv() << endl;


  return 0;
}


