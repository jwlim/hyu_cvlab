
// featuretracker.h
//
// Authors:
// VINS-mono 2018, archive'18
// Coverted to rvslam by Euntae Hong
//

#ifndef _RVSLAM_FEATURE_TRACKER_H_
#define _RVSLAM_FEATURE_TRACKER_H_

#include <fstream>
#include <map>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Dense>
#include "image_pyramid.h"
#include "rvslam_common.h"
#include <opencv2/opencv.hpp>

//#include "feature.h"  // Feature detector/tracker interface.
#include "visual_odometer.h"

using namespace std;


void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);


namespace rvslam {

class FeatureTracker {
  public:
    FeatureTracker();
    FeatureTracker(const VisualOdometer::Calib calib, const bool equalize);

    void readImage(const cv::Mat &_img);
    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    bool inBorder(const cv::Point2f &pt);


    // Additional functions
    // For N points
    void MakeNormalizedPoints(vector<cv::Point2f>& pts,
                              vector<cv::Point2f>* un_pts);
    // For eigen vector
    void MakeNormalizedPoints(Eigen::Vector2d& pts,
                              Eigen::Vector3d* un_pts); 
    void UpdateFtid();

    cv::Mat mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    double cur_time;
    double prev_time;

    static int n_id;


  private:
    int cols_;
    int rows_; 

    // init
    bool equalize_;
    bool pub_this_frame_;
    VisualOdometer::Calib calib_;
};


}


#endif
