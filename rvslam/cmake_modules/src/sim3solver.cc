// sim3solver.cc
//
// Authur: Jongwoo Lim(jongwoo.lim@gmail.com)
// Authur: Eunate Hong(dragon1301@naver.com)

#include <math.h>
#include <cmath>
#include <stdarg.h>

#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "sim3solver.h"

#include "rvslam_profile.h"

using namespace std;

namespace rvslam {

// Define funcitons
void ComputeT(const Mat3X& p1, const Mat3X& p2);

void Centeroid(const Mat3X& pts, Mat3X* pts_res, Mat3X* mean) {
  const int n = pts.cols();
  pts_res->resize(3, n);
  (*mean) = pts.rowwise().sum() / n;
  for (int i = 0; i < n; ++i) {
    pts_res->col(i) = pts.col(i) - (*mean);
  }
}

void Sim3Solver::Init(const vector<cv::Mat>& pts1, const vector<cv::Mat>& pts2,
                      const double probability, const int mininliers,
                      const int maxiter,
                      const cv::Mat calib) {
  const int n = pts1.size();
  mvX3Dc1 = pts1;
  mvX3Dc2 = pts2;

  mRansacProb = probability;
  mRansacMinInliers = mininliers;

  mvbInliersi.resize(n);
  float epsilon = (float)mRansacMinInliers / n;
  int niter;
  if (mRansacMinInliers == n)
    niter = 1;
  else
    niter = ceil(log(1-mRansacProb)/log(1-pow(epsilon,3)));

   mRansacMaxIts = max(1, min(niter, mRansacMaxIts));
   mnIterations = 0; 
   N = n;
   const double inlier_threshold = 5;
   mvnMaxError1.resize(n);
   mvnMaxError2.resize(n);

   for (int i = 0 ; i < n; ++i) {
     mvAllIndices.push_back(i);
     mvnIndices1.push_back(i);
     mvnMaxError1[i] = inlier_threshold;
     mvnMaxError2[i] = inlier_threshold;
   }

   mK1 = calib;
   mK2 = calib;

   FromCameraToImage(mvX3Dc1, mvP1im1, mK1);
   FromCameraToImage(mvX3Dc2, mvP2im2, mK2);

   mnBestInliers = 0;
}

void ComputeT(const Mat3X& p1, const Mat3X& p2) {
  // Horn 1987, Closed-form solution of absolute orientation using unit quaternions
  // Step 1: Centroid and relative coordinates
  const int n = p1.cols();
  const int m = p2.cols();
  Mat3X pr1(3,n), pr2(3,m);
  Mat3X o1(3,1), o2(3,1);
  Centeroid(p1, &pr1, &o1);
  Centeroid(p2, &pr2, &o2);

  // Step 2: Compute matrix a 
  Mat a = pr2 * pr1.transpose(); 

  // Step 3: Compute matrix b
  double n11, n12, n13, n14, n22, n23, n24, n33, n34, n44;
  Mat b(4,4);

  n11 = a(0,0) + a(1,1) + a(2,2);
  n12 = a(1,2) - a(2,1);
  n13 = a(2,0) - a(0,2);
  n14 = a(0,1) - a(1,0);
  n22 = a(0,0) - a(1,1) - a(2,2);
  n23 = a(0,1) + a(1,0);
  n24 = a(2,0) + a(0,2);
  n33 = -a(0,0) + a(1,1) - a(2,2);
  n34 = a(1,2) + a(2,1);
  n44 = -a(0,0) - a(1,1) + a(2,2);

  b << n11, n12, n13, n14,
    n12, n22, n23, n24,
    n13, n23, n33, n34,
    n14, n24, n34, n44;

  // Step4: Calculate highest eigenvalue
  Eigen::MatrixXcd eval, evec;
  Eigen::EigenSolver<Eigen::MatrixXd> es(b);
  eval = es.eigenvalues();
  evec = es.eigenvectors();

  Eigen::MatrixXd vec;
  vec = evec.col(0).real();
  const double norm = vec.norm();
  const double evec_0 = evec(0).real();
  double ang = atan2(norm, evec_0);
  vec = 2 * ang * vec/norm;

  Mat mr12(3,3);
  ceres::AngleAxisToRotationMatrix(vec.data(), mr12.data());

  // Step5: rotate
  Mat3X p3 = mr12 * pr2;


  // Step6: scale

  double nom = 0;
  for (int i = 0; i < n; i++) {
    Vec p1_vec = pr1.col(i);
    Vec p2_vec = p3.col(i);
    double s = p1_vec.dot(p2_vec);
    nom += s;


  }

  Mat3X aux_p3(3,m);
  aux_p3 = p3.array().pow(2);
  const double den = p3.sum();
  const double ms12 = nom / den;

  // Step 7: Translation
  Vec3 mt12;
  //mt12 = 1 - ms12 * mr12 * 2;

  // Step 8: Transformation
  //Mat mt12i(4,4);
  //Mat sr = ms12 * mr12;

}

// TODO(Euntae): This functions only test for debug
// It will be deleted after debug
void centroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
  cv::reduce(P,C,1,CV_REDUCE_SUM);
  C = C/P.cols;

  for(int i=0; i<P.cols; i++)
  {
    Pr.col(i)=P.col(i)-C;
  }
}

// TODO(Euntae): This functions only test for debug
// It will be deleted after debug
void Sim3Solver::ComputeTOpenCV(cv::Mat &P1, cv::Mat &P2) {
  // Custom implementation of:
  // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

  // Step 1: Centroid and relative coordinates

  cv::Mat Pr1(P1.size(),P1.type()); // Relative coordinates to centroid (set 1)
  cv::Mat Pr2(P2.size(),P2.type()); // Relative coordinates to centroid (set 2)
  cv::Mat O1(3,1,Pr1.type()); // Centroid of P1
  cv::Mat O2(3,1,Pr2.type()); // Centroid of P2

  centroid(P1,Pr1,O1);
  centroid(P2,Pr2,O2);

  // Step 2: Compute M matrix

  cv::Mat M = Pr2*Pr1.t();

  // Step 3: Compute N matrix

  double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

  cv::Mat N(4,4,P1.type());

  N11 = M.at<float>(0,0)+M.at<float>(1,1)+M.at<float>(2,2);
  N12 = M.at<float>(1,2)-M.at<float>(2,1);
  N13 = M.at<float>(2,0)-M.at<float>(0,2);
  N14 = M.at<float>(0,1)-M.at<float>(1,0);
  N22 = M.at<float>(0,0)-M.at<float>(1,1)-M.at<float>(2,2);
  N23 = M.at<float>(0,1)+M.at<float>(1,0);
  N24 = M.at<float>(2,0)+M.at<float>(0,2);
  N33 = -M.at<float>(0,0)+M.at<float>(1,1)-M.at<float>(2,2);
  N34 = M.at<float>(1,2)+M.at<float>(2,1);
  N44 = -M.at<float>(0,0)-M.at<float>(1,1)+M.at<float>(2,2);

  N = (cv::Mat_<float>(4,4) << N11, N12, N13, N14,
      N12, N22, N23, N24,
      N13, N23, N33, N34,
      N14, N24, N34, N44);


  // Step 4: Eigenvector of the highest eigenvalue

  cv::Mat eval, evec;

  cv::eigen(N,eval,evec); //evec[0] is the quaternion of the desired rotation

  cv::Mat vec(1,3,evec.type());
  (evec.row(0).colRange(1,4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

  // Rotation angle. sin is the norm of the imaginary part, cos is the real part
  double ang=atan2(norm(vec),evec.at<float>(0,0));

  vec = 2*ang*vec/norm(vec); //Angle-axis representation. quaternion angle is the half

  mR12i.create(3,3,P1.type());

  cv::Rodrigues(vec,mR12i); // computes the rotation matrix from angle-axis

  // Step 5: Rotate set 2

  cv::Mat P3 = mR12i*Pr2;

  // Step 6: Scale

  double nom = Pr1.dot(P3);
  cv::Mat aux_P3(P3.size(),P3.type());
  aux_P3=P3;
  cv::pow(P3,2,aux_P3);
  double den = 0;

  for(int i=0; i<aux_P3.rows; i++)
  {
    for(int j=0; j<aux_P3.cols; j++)
    {
      den+=aux_P3.at<float>(i,j);
    }
  }

  ms12i = nom/den;

  // Step 7: Translation

  mt12i.create(1,3,P1.type());
  mt12i = O1 - ms12i*mR12i*O2;

  // Step 8: Transformation

  // Step 8.1 T12
  mT12i = cv::Mat::eye(4,4,P1.type());

  cv::Mat sR = ms12i*mR12i;

  sR.copyTo(mT12i.rowRange(0,3).colRange(0,3));
  mt12i.copyTo(mT12i.rowRange(0,3).col(3));

  // Step 8.2 T21

  mT21i = cv::Mat::eye(4,4,P1.type());

  cv::Mat sRinv = (1.0/ms12i)*mR12i.t();

  sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
  cv::Mat tinv = -sRinv*mt12i;
  tinv.copyTo(mT21i.rowRange(0,3).col(3));
}

void Sim3Solver::CheckInliers() {
  vector<cv::Mat> vP1im2, vP2im1;
  Project(mvX3Dc2,vP2im1,mT12i,mK1);
  Project(mvX3Dc1,vP1im2,mT21i,mK2);

  mnInliersi=0;

  for(size_t i=0; i<mvP1im1.size(); i++)
  {
    cv::Mat dist1 = mvP1im1[i]-vP2im1[i];
    cv::Mat dist2 = vP1im2[i]-mvP2im2[i];

    float err1 = dist1.dot(dist1);
    float err2 = dist2.dot(dist2);

    if(err1<mvnMaxError1[i] && err2<mvnMaxError2[i])
    {
      mvbInliersi[i]=true;
      mnInliersi++;
    }
    else
      mvbInliersi[i]=false;
  }
}

void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw,
                         vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
  cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
  cv::Mat tcw = Tcw.rowRange(0,3).col(3);
  float fx = K.at<float>(0,0);
  float fy = K.at<float>(1,1);
  float cx = K.at<float>(0,2);
  float cy = K.at<float>(1,2);

  vP2D.clear();
  vP2D.reserve(vP3Dw.size());

  for(size_t i=0, iend=vP3Dw.size(); i<iend; i++)
  {
    cv::Mat P3Dc = Rcw*vP3Dw[i]+tcw;
    float invz = 1/(P3Dc.at<float>(2));
    float x = P3Dc.at<float>(0)*invz;
    float y = P3Dc.at<float>(1)*invz;

    vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
  }
}

cv::Mat Sim3Solver::EstimateSim3OpenCV(int nIterations,
    bool &bNoMore, vector<bool> vbInliers, int nInliers) {
  bNoMore = false;
  vbInliers = vector<bool>(mN1,false);
  nInliers=0;

  if(N<mRansacMinInliers)
  {
    bNoMore = true;
    return cv::Mat();
  }

  vector<size_t> vAvailableIndices;

  cv::Mat P3Dc1i(3,3,CV_32F);
  cv::Mat P3Dc2i(3,3,CV_32F);

  int nCurrentIterations = 0;
  while(mnIterations<mRansacMaxIts && nCurrentIterations<nIterations)
  {
    nCurrentIterations++;
    mnIterations++;

    vAvailableIndices = mvAllIndices;


    // Get min set of points
    for(short i = 0; i < 3; ++i)
    {
      //int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);
      int randi = rand() % vAvailableIndices.size()-1;

      int idx = vAvailableIndices[randi];

      mvX3Dc1[idx].copyTo(P3Dc1i.col(i));
      mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

      vAvailableIndices[idx] = vAvailableIndices.back();
      vAvailableIndices.pop_back();
    }

    ComputeTOpenCV(P3Dc1i,P3Dc2i);

    cout << "A" << endl;
    CheckInliers();
    cout << "B" << endl;

    if(mnInliersi>=mnBestInliers)
    {
      mvbBestInliers = mvbInliersi;
      mnBestInliers = mnInliersi;
      mBestT12 = mT12i.clone();
      mBestRotation = mR12i.clone();
      mBestTranslation = mt12i.clone();
      mBestScale = ms12i;

      if(mnInliersi>mRansacMinInliers)
      {
        nInliers = mnInliersi;
        for(int i=0; i<N; i++)
          if(mvbInliersi[i])
            vbInliers[mvnIndices1[i]] = true;
        return mBestT12;
      }
    }
  }


  if(mnIterations>=mRansacMaxIts)
    bNoMore=true;

  return cv::Mat();
}

void Sim3Solver::ComputeError(const Mat3X& pts0, const Mat3X& pts1,
                             const Mat3& em, Mat1X* err) {
  int n = err->cols();
  //double t = inlier_threshold_ * inlier_threshold_;
  double t = 3;
  for (int i = 0; i < n; i++) {
    Vec3 x1 = pts0.col(i);
    Vec3 x2 = pts1.col(i);
    Vec3 emx1 = em * x1;
    Vec3 emtx2 = em.transpose() * x2;
    double x2temx1 = x2.dot(emx1);

    double a = emx1[0] * emx1[0];
    double b = emx1[1] * emx1[1];
    double c = emtx2[0] * emtx2[0];
    double d = emtx2[1] * emtx2[1];
    double e = x2temx1 * x2temx1 / (a + b + c + d);

    (*err)(i) = (e > t)? t: e;
  }
}

void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
  float fx = K.at<float>(0,0);
  float fy = K.at<float>(1,1);
  float cx = K.at<float>(0,2);
  float cy = K.at<float>(1,2);

  vP2D.clear();
  vP2D.reserve(vP3Dc.size());

  for(size_t i=0, iend=vP3Dc.size(); i<iend; i++)
  {
    float invz = 1/(vP3Dc[i].at<float>(2));
    float x = vP3Dc[i].at<float>(0)*invz;
    float y = vP3Dc[i].at<float>(1)*invz;

    vP2D.push_back((cv::Mat_<float>(2,1) << fx*x+cx, fy*y+cy));
  }
}


}  // namespace rvslam
