// sim3solver.h
// 
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
// Author: Euntae Hong(dragon1301@naver.com)

#include <fstream>
#include <map>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Dense>
#include "ransac.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef _RVSLAM_SIM3_SOLVER_H_
#define _RVSLAM_SIM3_SOLVER_H_

namespace rvslam {

void ComputeTOpenCV(cv::Mat &P1, cv::Mat &P2);

class Sim3Solver {
 public:

  void Init(const std::vector<cv::Mat>& pts1, const std::vector<cv::Mat>& pts2,
            const double probability, const int mininliers,
            const int maxiter, const cv::Mat calib);

  //for debug
  cv::Mat EstimateSim3OpenCV(int nIterations,
                          bool &bNoMore, std::vector<bool> vbInliers, int nInliers); 
  void Project(const std::vector<cv::Mat> &vP3Dw, std::vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K);
  void CheckInliers();

 private:
  void ComputeError(const Mat3X&, const Mat3X&, const Mat3&, Mat1X*);

  //for debug
  void ComputeTOpenCV(cv::Mat &P1, cv::Mat &P2); 
  void FromCameraToImage(const std::vector<cv::Mat> &vP3Dc, std::vector<cv::Mat> &vP2D, cv::Mat K);
  int mnIterations;
  std::vector<bool> mvbBestInliers;
  int mnBestInliers;
  cv::Mat mBestT12;
  cv::Mat mBestRotation;
  cv::Mat mBestTranslation;
  float mBestScale;
  std::vector<size_t> mvAllIndices;
  double mRansacProb;
  // RANSAC min inliers
  int mRansacMinInliers;
  //
  // RANSAC max iterations
  int mRansacMaxIts;
  std::vector<cv::Mat> mvX3Dc1;
  std::vector<cv::Mat> mvX3Dc2;
  //std::vector<MapPoint*> mvpMapPoints1;
  //std::vector<MapPoint*> mvpMapPoints2;
  //std::vector<MapPoint*> mvpMatches12;
  std::vector<size_t> mvnIndices1;
  std::vector<size_t> mvSigmaSquare1;
  std::vector<size_t> mvSigmaSquare2;
  std::vector<size_t> mvnMaxError1;
  std::vector<size_t> mvnMaxError2;


  cv::Mat mR12i;
  cv::Mat mt12i;
  float ms12i;
  cv::Mat mT12i;
  cv::Mat mT21i;
  std::vector<bool> mvbInliersi;
  int mnInliersi;

  cv::Mat mK1, mK2;

  cv::vector<cv::Mat> mvP1im1;
  cv::vector<cv::Mat> mvP2im2;
  int N, mN1;

};


}  // namespace rvslam

#endif  // _RVSLAM_SIM3_SOLVER_H_
