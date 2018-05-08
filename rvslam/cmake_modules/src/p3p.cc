// p3p.cc
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//
// ComputePoseP3P from Laurent Kneip's P3P solver.
// RobustEstimatePoseP3P from libmv's robust_estimation.h.

#include "p3p.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>  // setfill(' ')
#include <glog/logging.h>

#include "rvslam_util.h"

using namespace std;

namespace rvslam {

namespace {

template<typename Real>
int SolveQuarticPolynomial(const Real *coeffs, Real *solutions) {
  Real A = coeffs[0];
  Real B = coeffs[1];
  Real C = coeffs[2];
  Real D = coeffs[3];
  Real E = coeffs[4];

  Real A_pw2 = A * A;
  Real B_pw2 = B * B;
  Real A_pw3 = A_pw2 * A;
  Real B_pw3 = B_pw2 * B;
  Real A_pw4 = A_pw3 * A;
  Real B_pw4 = B_pw3 * B;

  Real alpha = -3 * B_pw2 / (8 * A_pw2) + C / A;
  Real beta = B_pw3 / (8 * A_pw3) - B * C / (2 * A_pw2) + D / A;
  Real gamma = -3 * B_pw4 / (256 * A_pw4) + B_pw2 * C / (16 * A_pw3)
               - B * D / (4 * A_pw2) + E / A;
  Real alpha_pw2 = alpha * alpha;
  Real alpha_pw3 = alpha_pw2 * alpha;

  typedef std::complex<Real> Complex;
  Complex P(-alpha_pw2 / 12 - gamma, 0);
  Complex Q(-alpha_pw3 / 108 + alpha * gamma / 3 - pow(beta, 2) / 8, 0);
  Complex R = -Q / 2.0 + sqrt(pow(Q, 2.0) / 4.0 + pow(P, 3.0) / 27.0);
  Complex U = pow(R, (1.0 / 3.0));
  Complex y = U.real() == 0 ? -5.0 * alpha / 6.0 - pow(Q, (1.0 / 3.0)) :
                              -5.0 * alpha / 6.0 - P / (3.0 * U) + U;
  Complex w = sqrt(alpha + 2.0 * y);

  Complex temp;
  temp = -B / (4.0 * A)
      + 0.5 * (w + sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
  solutions[0] = temp.real();
  temp = -B / (4.0 * A)
      + 0.5 * (w - sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
  solutions[1] = temp.real();
  temp = -B / (4.0 * A)
      + 0.5 * (-w + sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));
  solutions[2] = temp.real();
  temp = -B / (4.0 * A)
      + 0.5 * (-w - sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));
  solutions[3] = temp.real();
  return 4;
}

template <typename T>
void RandomPermuteFirstN(int n, vector<T>* samples) {
  CHECK_NOTNULL(samples);
  if (n >= samples->size()) n = samples->size() - 1;
  for (int i = 0; i < n; ++i) {
    int j = i + (rand() % (samples->size() - i));
    swap(samples->at(i), samples->at(j));
  }
}

template <typename MatType, typename Idx>
MatType GetColumns(const MatType& mat, const vector<Idx>& idx) {
  MatType ret(mat.rows(), idx.size());
  for (int i = 0; i < idx.size(); ++i) ret.col(i) = mat.col(idx[i]);
  return ret;
}

}  // namespace

int ComputePosesP3P(const Mat3X& image_points, const Mat3X& world_points,
                    vector<Mat34>* solutions) {
  // Extraction of world points
  Vec3 P1 = world_points.col(0);
  Vec3 P2 = world_points.col(1);
  Vec3 P3 = world_points.col(2);

  // Verification that world points are not colinear
  if ((P2 - P1).cross(P3 - P1).norm() < 1e-9) return -1;

  // Extraction of feature vectors
  Vec3 f1 = image_points.col(0).normalized();
  Vec3 f2 = image_points.col(1).normalized();
  Vec3 f3 = image_points.col(2).normalized();

  // Creation of intermediate camera frame
  Vec3 e1 = f1;
  Vec3 e3 = f1.cross(f2).normalized();
  Vec3 e2 = e3.cross(e1);

  Mat3 T;
  T.row(0) = e1;
  T.row(1) = e2;
  T.row(2) = e3;
  f3 = T * f3;

  // Reinforce that f3[2] > 0 for having theta in [0;pi]
  if (f3(2) > 0) {
    f1 = image_points.col(1).normalized();
    f2 = image_points.col(0).normalized();
    f3 = image_points.col(2).normalized();

    e1 = f1;
    e3 = f1.cross(f2).normalized();
    e2 = e3.cross(e1);

    T.row(0) = e1;
    T.row(1) = e2;
    T.row(2) = e3;
    f3 = T * f3;

    P1 = world_points.col(1);
    P2 = world_points.col(0);
    P3 = world_points.col(2);
  }

  // Creation of intermediate world frame
  Vec3 n1 = (P2 - P1).normalized();
  Vec3 n3 = n1.cross(P3 - P1).normalized();
  Vec3 n2 = n3.cross(n1);

  Mat3 N;
  N.row(0) = n1;
  N.row(1) = n2;
  N.row(2) = n3;

  // Extraction of known parameters
  P3 = N * (P3 - P1);

  double d_12 = (P2 - P1).norm();
  double f_1 = f3(0) / f3(2);
  double f_2 = f3(1) / f3(2);
  double p_1 = P3(0);
  double p_2 = P3(1);

  double cos_beta = f1.dot(f2);
  double b = 1 / (1 - pow(cos_beta, 2)) - 1;
  b = (cos_beta < 0) ? -sqrt(b) : sqrt(b);

  // Definition of temporary variables for avoiding multiple computation
  double f_1_pw2 = pow(f_1, 2);
  double f_2_pw2 = pow(f_2, 2);
  double p_1_pw2 = pow(p_1, 2);
  double p_1_pw3 = p_1_pw2 * p_1;
  double p_1_pw4 = p_1_pw3 * p_1;
  double p_2_pw2 = pow(p_2, 2);
  double p_2_pw3 = p_2_pw2 * p_2;
  double p_2_pw4 = p_2_pw3 * p_2;
  double d_12_pw2 = pow(d_12, 2);
  double b_pw2 = pow(b, 2);

  // Computation of factors of 4th degree polynomial
  Eigen::Matrix<double, 5, 1> factors;
  factors(0) =  -f_2_pw2 * p_2_pw4
               - p_2_pw4 * f_1_pw2
               - p_2_pw4;
  factors(1) =   2 * p_2_pw3 * d_12 * b
               + 2 * f_2_pw2 * p_2_pw3 * d_12 * b
               - 2 * f_2 * p_2_pw3 * f_1 * d_12;
  factors(2) =  -f_2_pw2 * p_2_pw2 * p_1_pw2
               - f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2
               - f_2_pw2 * p_2_pw2 * d_12_pw2
               + f_2_pw2 * p_2_pw4
               + p_2_pw4 * f_1_pw2
               + 2 * p_1 * p_2_pw2 * d_12
               + 2 * f_1 * f_2 * p_1 * p_2_pw2 * d_12 * b
               - p_2_pw2 * p_1_pw2 * f_1_pw2
               + 2 * p_1 * p_2_pw2 * f_2_pw2 * d_12
               - p_2_pw2 * d_12_pw2 * b_pw2
               - 2 * p_1_pw2 * p_2_pw2;
  factors(3) =   2 * p_1_pw2 * p_2 * d_12 * b
               + 2 * f_2 * p_2_pw3 * f_1 * d_12
               - 2 * f_2_pw2 * p_2_pw3 * d_12 * b
               - 2 * p_1 * p_2 * d_12_pw2 * b;
  factors(4) =  -2 * f_2 * p_2_pw2 * f_1 * p_1 * d_12 * b
               + f_2_pw2 * p_2_pw2 * d_12_pw2
               + 2 * p_1_pw3 * d_12
               - p_1_pw2 * d_12_pw2
               + f_2_pw2 * p_2_pw2 * p_1_pw2
               - p_1_pw4
               - 2 * f_2_pw2 * p_2_pw2 * p_1 * d_12
               + p_2_pw2 * f_1_pw2 * p_1_pw2
               + f_2_pw2 * p_2_pw2 * d_12_pw2 * b_pw2;
  // Computation of roots
  Vec4 roots;
  SolveQuarticPolynomial(factors.data(), roots.data());

  // Backsubstitution of each solution
  solutions->clear();
  for (int i = 0; i < 4; i++) {
    double cot_alpha = (-f_1 * p_1 / f_2 - roots(i) * p_2 + d_12 * b)
        / (-f_1 * roots(i) * p_2 / f_2 + p_1 - d_12);

    double cos_theta = roots(i);
    double sin_theta = sqrt(1 - pow(roots(i), 2));
    double sin_alpha = sqrt(1 / (pow(cot_alpha, 2) + 1));
    double cos_alpha = sqrt(1 - pow(sin_alpha, 2));
    if (cot_alpha < 0) cos_alpha = -cos_alpha;

    Vec3 C;
    C << d_12 * cos_alpha * (sin_alpha * b + cos_alpha),
         cos_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha),
         sin_theta * d_12 * sin_alpha * (sin_alpha * b + cos_alpha);
    C = P1 + N.transpose() * C;

    Mat3 R;
    R << -cos_alpha, -sin_alpha * cos_theta, -sin_alpha * sin_theta,
         sin_alpha, -cos_alpha * cos_theta, -cos_alpha * sin_theta,
         0, -sin_theta, cos_theta;
    R = N.transpose() * R.transpose() * T;

    Mat34 solution;
    solution.block(0, 0, 3, 3) = R.transpose();
    solution.block(0, 3, 3, 1) = -R.transpose() * C;
    solutions->push_back(solution);
  }
  return 0;
}

bool RobustEstimatePoseP3P(const Mat3X& image_points,
                           const Mat3X& world_points,
                           double inlier_threshold,
                           Mat34* best_model_ret,
                           vector<int> *best_inliers_ret,
                           double *best_score_ret,
                           double failure_probability,
                           size_t max_iterations) {
  CHECK_NOTNULL(best_model_ret);
  CHECK_LT(failure_probability, 1.0);
  CHECK_GT(failure_probability, 0.0);
  CHECK_EQ(image_points.cols(), world_points.cols());
  const size_t min_samples = 3;
  const size_t total_samples = image_points.cols();
  const double inlier_threshold_sqr = inlier_threshold * inlier_threshold;
  // Test if we have sufficient points to for the kernel.
  if (total_samples < min_samples) return false;

  double best_cost = HUGE_VAL;
  vector<int> best_inliers;
  Mat34 best_model;
  size_t num_iterations = max_iterations;

  // In this robust estimator, the scorer always works on all the data points
  // at once. So precompute the list ahead of time.
  vector<int> all_samples(total_samples);
  for (int i = 0; i < total_samples; ++i) all_samples[i] = i;

  vector<int> samples;
  vector<Mat34> models;
  for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
    samples = all_samples;
    RandomPermuteFirstN(min_samples, &samples);
    samples.resize(min_samples);

    // Fit models.
    Mat3X x = GetColumns(image_points, samples);
    Mat3X X = GetColumns(world_points, samples);
    ComputePosesP3P(x, X, &models);

    // Compute costs for each model.
    vector<int> inliers;
    inliers.reserve(total_samples);
    for (int i = 0; i < models.size(); ++i) {
      const Mat34& model = models[i];
      // Compute inliers and cost of the model.
      double cost = 0.0;
      inliers.clear();
      for (int j = 0; j < total_samples; ++j) {
        Vec3 diff = Project(model, world_points.col(j)) - image_points.col(j);
        double err = diff.squaredNorm();
        if (err < inlier_threshold_sqr) {
          cost += err;
          inliers.push_back(j);
        } else {
          cost += inlier_threshold;
        }
      }
      if (cost < best_cost) {
        best_cost = cost;
        best_inliers = inliers;
        best_model = model;
        if (inliers.empty() == false) {
          double inlier_ratio = inliers.size() / (double) total_samples;
          num_iterations = static_cast<unsigned int>(
              log(failure_probability) /
              log(1.0 - pow(inlier_ratio, (double) min_samples)));
          if (num_iterations > max_iterations) num_iterations = max_iterations;
        }
      }
    }
  }
  //LOG(INFO) << setfill(' ') << "best_model: " << best_cost << ", "
  //    << best_inliers.size() << "/" << total_samples << ", "
  //   << num_iterations << " iter.\n" << best_model;
  if (best_model_ret) *best_model_ret = best_model;
  if (best_inliers_ret) *best_inliers_ret = best_inliers;
  if (best_score_ret) *best_score_ret = best_cost;
  return !best_inliers.empty();
}

}  // namespace rvslam

