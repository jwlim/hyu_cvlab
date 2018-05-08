// ransac.h
//
// Author: Po-Chen Wu (pcwu0329@gmail.com)
//         Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_RANSAC_H_
#define _RVSLAM_RANSAC_H_

#include <vector>
#include <Eigen/Dense>
#include "rvslam_common.h"

namespace rvslam {

class RanSaC {
 protected:
  // For nonlinear optimization in Eigen  
  template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
  struct Functor {
    typedef _Scalar Scalar;
    enum {
      InputsAtCompileTime = NX,
      ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime>
              JacobianType;
    
    int m_inputs, m_values;
    
    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
    
    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
  };

  RanSaC(int model_points) : model_points_(model_points) {
    inlier_threshold_ = 0.01;
    failure_probability_ = 1e-3;
    max_iterations_ = 100;
  }

  RanSaC(int model_points, double inlier_threshold, double failure_probability,
         int max_iterations) :
      model_points_(model_points), 
      inlier_threshold_(inlier_threshold),
      failure_probability_(failure_probability),
      max_iterations_(max_iterations) {}
 
  bool GetSubset2D2D(const Mat3X&, const Mat3X&, Mat3X*, Mat3X*, int);

  bool GetSubset2D3D(const Mat3X&, const Mat3X&, Mat3X*, Mat3X*, int);

  bool GetSubset3D3D(const Mat3X&, const Mat3X&, const Mat3X&, const Mat3X&,
                     Mat3X*, Mat3X*, Mat3X*, Mat3X*, int);

  bool CheckSubset(const Mat3X&);
  
  int FindInliers(const Mat3X&, const Mat3X&, const Mat&, Mat1X*, Mask*);
  
  virtual void ComputeError(const Mat3X&, const Mat3X&, const Mat&,
                            Mat1X*) = 0;
  
  int UpdateRanSaCNumIters(double, int);

  int model_points_;
  double inlier_threshold_;
  double failure_probability_;
  int max_iterations_;
};

}  // namespace rvslam
#endif  // _RVSLAM_RANSAC_H_
