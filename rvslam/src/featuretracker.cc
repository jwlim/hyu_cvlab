
// featuretracker.cc
//
// Authors:
// VINS-mono 2018, archive'18
// Coverted to rvslam by Euntae Hong
// 

#include <math.h>
#include <stdarg.h>

#include <map>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "klt_tracker.h"
#include "image_pyramid.h"
#include "image_util.h"  // Convolve.

#include "rvslam_profile.h"

#include "featuretracker.h"

using namespace std;

DEFINE_int32(feat_max_cnt, 200, "Maximum number of keyframes in the map.");
DEFINE_double(feat_min_dist, 30, "Pose inlier threshold in pixels.");
DEFINE_double(feat_f_threshold, 1, "Pose inlier threshold in pixels.");

namespace rvslam { 
int FeatureTracker::n_id = 0;

bool FeatureTracker::inBorder(const cv::Point2f &pt) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < cols_ - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < rows_ - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}


FeatureTracker::FeatureTracker(const VisualOdometer::Calib calib, const bool equalize) {
  calib_ = calib;
  equalize_ = equalize;
  pub_this_frame_ = true;
}

void FeatureTracker::setMask() {
  mask = cv::Mat(rows_, cols_, CV_8UC1, cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int> > > cnt_pts_id;

  for (unsigned int i = 0; i < forw_pts.size(); i++)
    cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
      return a.first > b.first;
      });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, FLAGS_feat_min_dist, 0, -1);
    }
  }
}

void FeatureTracker::addPoints()
{
  for (auto &p : n_pts)
  {
    forw_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

void FeatureTracker::readImage(const cv::Mat &_img) {
  cols_ = _img.cols;
  rows_ = _img.rows;
  cv::Mat img;

  if (equalize_) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(_img, img);
  }
  else
    img = _img;

  if (forw_img.empty())
  {
    prev_img = cur_img = forw_img = img;
  }
  else
  {
    forw_img = img;
  }

  forw_pts.clear();

  if (cur_pts.size() > 0)
  {
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++)
      if (status[i] && !inBorder(forw_pts[i]))
        status[i] = 0;
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto &n : track_cnt)
    n++;

  if (pub_this_frame_) {
    rejectWithF();
    setMask();

    int n_max_cnt = FLAGS_feat_max_cnt - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0)
    {
      if(mask.empty())
        cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;
      if (mask.size() != forw_img.size())
        cout << "wrong size " << endl;
      cv::goodFeaturesToTrack(forw_img, n_pts, FLAGS_feat_max_cnt - forw_pts.size(), 0.01, FLAGS_feat_min_dist, mask);
    }
    else
      n_pts.clear();

    addPoints();
  }


  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img = forw_img;
  cur_pts = forw_pts;
  undistortedPoints();

  UpdateFtid();
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time) {
  cols_ = _img.cols;
  rows_ = _img.rows;
  cv::Mat img;
  cur_time = _cur_time;

  if (equalize_) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(_img, img);
  }
  else
    img = _img;

  if (forw_img.empty())
  {
    prev_img = cur_img = forw_img = img;
  }
  else
  {
    forw_img = img;
  }

  forw_pts.clear();

  if (cur_pts.size() > 0)
  {
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++)
      if (status[i] && !inBorder(forw_pts[i]))
        status[i] = 0;
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto &n : track_cnt)
    n++;

  if (pub_this_frame_) {
    rejectWithF();
    setMask();

    int n_max_cnt = FLAGS_feat_max_cnt - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0)
    {
      if(mask.empty())
        cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;
      if (mask.size() != forw_img.size())
        cout << "wrong size " << endl;
      cv::goodFeaturesToTrack(forw_img, n_pts, FLAGS_feat_max_cnt - forw_pts.size(), 0.01, FLAGS_feat_min_dist, mask);
    }
    else
      n_pts.clear();

    addPoints();
  }
  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img = forw_img;
  cur_pts = forw_pts;
  undistortedPoints();
  prev_time = cur_time;

  UpdateFtid();
}

void FeatureTracker::UpdateFtid() {
  for (unsigned int i = 0;; i++) {
    bool completed = false;
    completed |= updateID(i);
    if (!completed)
      break;
  }
}

void FeatureTracker::rejectWithF() {
  if (forw_pts.size() >= 8) {
    vector<cv::Point2f> un_cur_pts, un_forw_pts;
    MakeNormalizedPoints(cur_pts, &un_cur_pts);
    MakeNormalizedPoints(forw_pts, &un_forw_pts);

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC,
                           FLAGS_feat_f_threshold, 0.99, status);
    int size_a = cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
  }
}

void FeatureTracker::MakeNormalizedPoints(vector<cv::Point2f>& pts,
                                          vector<cv::Point2f>* un_pts) {
    const double fx = calib_.fx, fy = calib_.fy, cx = calib_.cx, cy = calib_.cy;
    const double k0 = calib_.k0, k1 = calib_.k1;
    const double k2 = calib_.k2;
    const double k3 = calib_.k3;
    const double k4 = calib_.k4;
    const int n = pts.size();
    un_pts->resize(n);

    for (int i = 0; i < n; ++i) {
      double nx = (pts[i].x - cx) / fx;
      double ny = (pts[i].y - cy) / fy;
      double nx_out, ny_out;
      if (k1 != 0.0 || k2 != 0.0) {
        double x = nx, y = ny;
        for (int i = 0; i < 10; i++) {
          const double x2 = pow(x,2), y2 = pow(y,2), xy = 2 * x * y, r2 = x2 + y2;
          const double rad = 1 + r2 * (k0 + r2 * (k1 + r2 * k4));
          const double ux = (nx - (xy * k2 + (r2 + 2 * x2) * k3)) / rad;
          const double uy = (ny - ((r2 + 2 * y2) * k2 + xy * k3)) / rad;
          const double dx = x - ux, dy = y - uy;
          x = ux,  y = uy;
          if (pow(dx,2) + pow(dy,2) < 1e-9) break;
        }
        nx = x, ny = y;
      }
      nx_out = nx, ny_out = ny;
      (*un_pts)[i].x = nx_out;
      (*un_pts)[i].y = ny_out;
    }
}

void FeatureTracker::MakeNormalizedPoints(Eigen::Vector2d& pts,
                                          Eigen::Vector3d* un_pts) {
    const double fx = calib_.fx, fy = calib_.fy, cx = calib_.cx, cy = calib_.cy;
    const double k0 = calib_.k0, k1 = calib_.k1;
    const double k2 = calib_.k2;
    const double k3 = calib_.k3;
    const double k4 = calib_.k4;
    const int n = pts.size();

    double nx = (pts(0) - cx) / fx;
    double ny = (pts(1) - cy) / fy;
    double nx_out, ny_out;
    if (k1 != 0.0 || k2 != 0.0) {
      double x = nx, y = ny;
      for (int i = 0; i < 10; i++) {
        const double x2 = pow(x,2), y2 = pow(y,2), xy = 2 * x * y, r2 = x2 + y2;
        const double rad = 1 + r2 * (k0 + r2 * (k1 + r2 * k4));
        const double ux = (nx - (xy * k2 + (r2 + 2 * x2) * k3)) / rad;
        const double uy = (ny - ((r2 + 2 * y2) * k2 + xy * k3)) / rad;
        const double dx = x - ux, dy = y - uy;
        x = ux,  y = uy;
        if (pow(dx,2) + pow(dy,2) < 1e-9) break;
      }
      nx = x, ny = y;
    }
    nx_out = nx, ny_out = ny;
    (*un_pts) << nx_out, ny_out, 1.0;
}

bool FeatureTracker::updateID(unsigned int i)
{
  if (i < ids.size())
  {
    if (ids[i] == -1)
      ids[i] = n_id++;
    return true;
  }
  else
    return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
  //m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name) {
  const double fx = calib_.fx;;
  const double fy = calib_.fy;

  cv::Mat undistortedImg(rows_ + 600, cols_ + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < cols_; i++)
    for (int j = 0; j < rows_; j++)
    {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      MakeNormalizedPoints(a, &b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
    }
  for (int i = 0; i < int(undistortedp.size()); i++)
  {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * fy + cols_ / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * fx + rows_ / 2;
    pp.at<float>(2, 0) = 1.0;
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < rows_ + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < cols_ + 600)
    {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
  cur_un_pts.clear();
  cur_un_pts_map.clear();
  //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
  for (unsigned int i = 0; i < cur_pts.size(); i++)
  {
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    MakeNormalizedPoints(a, &b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
  }

  // caculate points velocity
  if (!prev_un_pts_map.empty())
  {
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++)
    {
      if (ids[i] != -1)
      {
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end())
        {
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.push_back(cv::Point2f(v_x, v_y));
        }
        else
          pts_velocity.push_back(cv::Point2f(0, 0));
      }
      else
      {
        pts_velocity.push_back(cv::Point2f(0, 0));
      }
    }
  }
  else
  {
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}


}
