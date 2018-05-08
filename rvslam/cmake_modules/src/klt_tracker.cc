// klt_tracker.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
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
extern rvslam::ProfileDBType pdb_;

using namespace std;

DEFINE_int32(klt_num_levels, 3,
             "The number of image pyramid levels in KLT Tracker.");
DEFINE_double(klt_sigma, 1.0, "Gaussian sigma for KLT tracker.");
DEFINE_double(klt_min_cornerness, 50, "The mininum cornerness response.");
DEFINE_int32(klt_max_features, 300, "The maximum number of features to track.");
DEFINE_int32(klt_redetect_thr, 200, "The threshold to redetect features.");
DEFINE_int32(klt_num_loop, 10, "The number of loops in each iteration.");
DEFINE_int32(klt_detect_level, 0, "The pyramid level to detect features.");
DEFINE_int32(klt_stereo_min_match_score, 5,
             "The minimum match score per pixel (0-255)");
DEFINE_int32(klt_stereo_y_offset, 8,
             "The maximum y-offset in stereo feature matching.");
DEFINE_int32(klt_stereo_max_disp, 64,
             "The maximum disparity in stereo feature matching.");

namespace rvslam {

typedef Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> ArrayXXu8;
typedef Eigen::ArrayXXf ArrayF;

template <class T> inline
string ToString(const T& v) {
  stringstream ss;
  ss << v;
  return ss.str();
}

inline bool operator<(const KLTFeat& ft1, const KLTFeat& ft2) {
  return ft1.score < ft2.score;
}

template <typename T, int d, int e> inline
Eigen::Matrix<T, d + e, 1> Concat(const Eigen::Matrix<T, d, 1>& v1,
                                  const Eigen::Matrix<T, e, 1>& v2) {
  Eigen::Matrix<T, d + e, 1> ret;
  ret << v1, v2;
  return ret;
}

void FindCorners(const ImagePyramid::Level& level, int level_idx,
                 const ArrayF& gaussian_kernel, double min_cornerness,
                 vector<KLTFeat>* feats) {
  // Compute cornerness.
  const int rad = gaussian_kernel.rows() / 2;
  const int width = level.imgf.rows(), height = level.imgf.cols();
  ArrayF cornerness(width, height);
  ArrayF img_x2 = level.imgx * level.imgx;
  ArrayF img_xy = level.imgx * level.imgy;
  ArrayF img_y2 = level.imgy * level.imgy;
  ArrayF tmp(width, height);
  Convolve(img_x2, gaussian_kernel, &tmp);
  Convolve(tmp, gaussian_kernel.transpose(), &img_x2);
  Convolve(img_xy, gaussian_kernel, &tmp);
  Convolve(tmp, gaussian_kernel.transpose(), &img_xy);
  Convolve(img_y2, gaussian_kernel, &tmp);
  Convolve(tmp, gaussian_kernel.transpose(), &img_y2);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const double ix2 = img_x2(x, y), ixy = img_xy(x, y), iy2 = img_y2(x, y);
      const double p1 = ix2 + iy2;
      const double p2 = sqrt(4 * ixy * ixy + (ix2 - iy2) * (ix2 - iy2));
      cornerness(x, y) = 0.5 * min(p1 + p2, p1 - p2)
          + 1e-6 * rand() / RAND_MAX;
    }
  }
/*
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (x < rad || y < rad || x >= width - rad || y >= height - rad) {
        cornerness(x, y) = 1;
        continue;
      }
      double ix2 = 0.0, ixy = 0.0, iy2 = 0.0;
      for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
          const float ix = level.imgx(x + dx, y + dy);
          const float iy = level.imgy(x + dx, y + dy);
          const double w = gaussian_kernel(dx + rad, 0) *
                           gaussian_kernel(dy + rad, 0);
          ix2 += w * ix * ix;
          ixy += w * ix * iy;
          iy2 += w * iy * iy;
        }
      }
      const double p1 = ix2 + iy2;
      const double p2 = sqrt(4 * ixy * ixy + (ix2 - iy2) * (ix2 - iy2));
      cornerness(x, y) = 0.5 * min(p1 + p2, p1 - p2)
          + 1e-6 * rand() / RAND_MAX;
    }
  }
*/

  // Mark existing features.
  const double kExistingFeature = 1e9;
  const double level_scale = (1L << level_idx);
  for (int i = 0; i < feats->size(); ++i) {
    KLTFeat& ft = feats->at(i);
    int x = ft.pos[0] / level_scale, y = ft.pos[1] / level_scale;
    ft.score = cornerness(x, y);
    cornerness(x, y) = kExistingFeature;
  }
  // Detect new features after non-maximal suppresssion.
  const int margin = 2 * rad + 1;
  for (int y = margin; y < height - margin; ++y) {
    for (int x = margin; x < width - margin; ++x) {
      const double cornerness_x_y = cornerness(x, y);
      if (cornerness_x_y == kExistingFeature ||
          cornerness_x_y < min_cornerness) continue;
      bool local_max = true;
      for (int dy = -margin; dy <= margin; ++dy) {
        for (int dx = -margin; dx <= margin; ++dx) {
          if (cornerness(x + dx, y + dy) > cornerness_x_y) {
            local_max = false;
            break;
          }
        }
      }
      if (local_max) {
        Eigen::Vector2f pos(x * level_scale, y * level_scale);
        feats->push_back(KLTFeat(-1, level_idx, pos, cornerness_x_y));
      }
    }
  }
}

void SortAndFilterCorners(int num_features, int max_features,
                          vector<KLTFeat>* feats) {
  stable_sort(feats->begin() + num_features, feats->end());
  if (feats->size() > max_features) feats->resize(max_features);
}

size_t FilterDuplicateTracks(int rad, vector<KLTFeat>* feats) {
  multimap<float, int> ypos;
  for (int i = 0; i < feats->size(); ++i) {
    const Eigen::Vector2f& p = feats->at(i).pos;
    if (p[0] >= 0 && p[1] >= 0) ypos.insert(make_pair(p[1], i));
  }
  vector<KLTFeat> tmp_feats;
  tmp_feats.reserve(feats->size());
  for (int i = 0; i < feats->size(); ++i) {
    const KLTFeat& ft = feats->at(i);
    const int ft_x = ft.pos[0], ft_y = ft.pos[1];
    if (ft_x < 0 || ft_y < 0) continue;
    multimap<float, int>::const_iterator it, it_lb, it_ub;
    it_lb = ypos.lower_bound(ft_y - rad);
    it_ub = ypos.upper_bound(ft_y + rad);
    bool duplicate = false;
    for (it = it_lb; it != it_ub && !duplicate; ++it) {
      const KLTFeat& ft2 = feats->at(it->second);
      duplicate = (ft_x - rad < ft2.pos[0] && ft2.pos[0] < ft_x + rad &&
                   ft.score < ft2.score);
    }
    if (!duplicate) {
      tmp_feats.push_back(ft);
    }
  }
  feats->swap(tmp_feats);
  return feats->size();
}

size_t FilterDuplicateTracksIMU(int rad, vector<KLTFeat>* feats) {
  multimap<float, int> ypos;
  for (int i = 0; i < feats->size(); ++i) {
    const Eigen::Vector2f& p = feats->at(i).pos;
    if (p[0] >= 0 && p[1] >= 0) ypos.insert(make_pair(p[1], i));
  }
  vector<KLTFeat> tmp_feats;
  tmp_feats.reserve(feats->size());
  for (int i = 0; i < feats->size(); ++i) {
    const KLTFeat& ft = feats->at(i);
    const int ft_x = ft.pos[0], ft_y = ft.pos[1];
    if (ft_x < 0 || ft_y < 0) continue;
    multimap<float, int>::const_iterator it, it_lb, it_ub;
    it_lb = ypos.lower_bound(ft_y - rad);
    it_ub = ypos.upper_bound(ft_y + rad);
    bool duplicate = false;
    for (it = it_lb; it != it_ub && !duplicate; ++it) {
      const KLTFeat& ft2 = feats->at(it->second);
      duplicate = (ft_x - rad < ft2.pos[0] && ft2.pos[0] < ft_x + rad &&
                   ft.score < ft2.score);
    }
    if (!duplicate) {
      tmp_feats.push_back(ft);
    }
  }
  feats->swap(tmp_feats);
  return feats->size();
}

bool KLTDetectFeatures(const ImagePyramid& pyr, const ArrayF& gaussian_kernel,
                       int max_features, double min_cornerness,
                       vector<KLTFeat>* feats) {
  const int org_size = feats->size();
  const int rad = gaussian_kernel.rows() / 2;
  for (int l = 0; feats->size() < max_features && l < pyr.num_levels(); ++l) {
    const ImagePyramid::Level& level = pyr[l];
    FindCorners(level, l, gaussian_kernel, min_cornerness, feats);
    SortAndFilterCorners(org_size, max_features, feats);
  }
  for (int i = org_size; i < feats->size(); ++i) {
    KLTFeat& ft = feats->at(i);
    ft.score = 0.0;
  }
  return true;
}

double Distance(double x0, double y0, double x1, double y1) {
  const double dx = x0 - x1, dy = y0 - y1;
  return sqrt(dx * dx + dy * dy);
}

double Distance(const Vec3& pt1, const Vec3& pt2) {
  return Distance(pt1(0), pt1(1), pt2(0), pt2(1));
}

bool KLTTrackFeaturesIMU(const ImagePyramid& prev, const ImagePyramid& cur,
                      const ArrayF& gaussian_kernel, const int num_loop,
                         const Mat3 rot,
                      vector<KLTFeat>* feats) {
  if (feats->size() <= 0) return true;
  // Track features from the top level to the bottom.
  const int rad = gaussian_kernel.rows() / 2;
  const int win_size = (2 * rad + 1);
  ArrayF patch_dx(win_size, win_size), patch_dy(win_size, win_size);
  ArrayF patch_dt(win_size, win_size);
  ArrayF patch0(win_size, win_size);
  ArrayF patch1_dx(win_size, win_size);
  ArrayF patch1_dy(win_size, win_size);
  ArrayF patch1_dt(win_size, win_size);
  for (int i = 0; i < feats->size(); ++i) {
    const int num_levels = cur.num_levels();
    const double pow2level = (1L << num_levels);
    KLTFeat& ft = feats->at(i);
    double x = ft.pos[0] / pow2level;
    double y = ft.pos[1] / pow2level;
    double x0 = x, y0 = y;
    double score = 0.0;
    for (int l = num_levels - 1; l >= 0; --l) {
      x *= 2, y *= 2, x0 *= 2, y0 *= 2;
      if (l < ft.level) continue;
      const ImagePyramid::Level& level_prev = prev[l];
      const ImagePyramid::Level& level_cur = cur[l];
      const int width = level_cur.imgf.rows(), height = level_cur.imgf.cols();
      const int x_max = width - rad - 1, y_max = height - rad - 1;
      Interp2Patch(level_prev.imgx, x0, y0, &patch1_dx);
      Interp2Patch(level_prev.imgy, x0, y0, &patch1_dy);
      Interp2Patch(level_prev.imgf, x0, y0, &patch1_dt);
patch1_dt -= patch1_dt.sum() / patch1_dt.size();
      for (int loop = 0; loop < num_loop; ++loop) {
        if (!(rad <= x && rad <= y && x < x_max && y < y_max)) {
          x = -1.f, y = -1.f;  // Set invalid coordinates.
          break;
        }
        double e[2] = { 0, 0 }, Z[3] = { 0, 0, 0 };
        score = 0.0;

//        LOG(INFO) << "interp2 begin";
        Interp2Patch(level_cur.imgx, x, y, &patch0);
        patch_dx = (patch0 + patch1_dx) / 2;
        Interp2Patch(level_cur.imgy, x, y, &patch0);
        patch_dy = (patch0 + patch1_dy) / 2;
        Interp2Patch(level_cur.imgf, x, y, &patch0);
patch0 -= patch0.sum() / patch0.size();
        patch_dt = patch0 - patch1_dt;
        Z[0] -= patch_dx.square().sum();
        Z[1] -= (patch_dx * patch_dy).sum();
        Z[2] -= patch_dy.square().sum();
        e[0] += (patch_dx * patch_dt).sum();
        e[1] += (patch_dy * patch_dt).sum();
        score += patch_dt.abs().sum();
        const double coef = 1.0 / (Z[0] * Z[2] - Z[1] * Z[1] + 1e-9);
        const double Z_inv[3] = { coef * Z[2], -coef * Z[1], coef * Z[0] };
        const double dx = Z_inv[0] * e[0] + Z_inv[1] * e[1];
        const double dy = Z_inv[1] * e[0] + Z_inv[2] * e[1];
        x -= dx * (loop + 1) / num_loop;
        y -= dy * (loop + 1) / num_loop;
      }
      if (!(rad <= x && rad <= y && x < x_max && y < y_max)) {
        x = -1.f, y = -1.f;
        break;
      }
    }

    Vec3 p_prev, p1, p2;
    p_prev << ft.pos[0], ft.pos[1], 1; 

    ft.pos << x, y;
    ft.score = score;

//Check predicted radius
    p1 << x, y, 1;
    p2 = rot * p_prev;
    p2 = p2 / p2(2);

    double dist = Distance(p1, p2);
    if (dist > 30) 
        ft.pos << -1.f, -1.f;
  }

  return true;
}


bool KLTTrackFeatures(const ImagePyramid& prev, const ImagePyramid& cur,
                      const ArrayF& gaussian_kernel, const int num_loop,
                      vector<KLTFeat>* feats) {
  if (feats->size() <= 0) return true;
  // Track features from the top level to the bottom.
  const int rad = gaussian_kernel.rows() / 2;
  const int win_size = (2 * rad + 1);
  ArrayF patch_dx(win_size, win_size), patch_dy(win_size, win_size);
  ArrayF patch_dt(win_size, win_size);
  ArrayF patch0(win_size, win_size);
  ArrayF patch1_dx(win_size, win_size);
  ArrayF patch1_dy(win_size, win_size);
  ArrayF patch1_dt(win_size, win_size);
  for (int i = 0; i < feats->size(); ++i) {
    const int num_levels = cur.num_levels();
    const double pow2level = (1L << num_levels);
    KLTFeat& ft = feats->at(i);
    double x = ft.pos[0] / pow2level;
    double y = ft.pos[1] / pow2level;
    double x0 = x, y0 = y;
    double score = 0.0;
    for (int l = num_levels - 1; l >= 0; --l) {
      x *= 2, y *= 2, x0 *= 2, y0 *= 2;
      if (l < ft.level) continue;
      const ImagePyramid::Level& level_prev = prev[l];
      const ImagePyramid::Level& level_cur = cur[l];
      const int width = level_cur.imgf.rows(), height = level_cur.imgf.cols();
      const int x_max = width - rad - 1, y_max = height - rad - 1;
      Interp2Patch(level_prev.imgx, x0, y0, &patch1_dx);
      Interp2Patch(level_prev.imgy, x0, y0, &patch1_dy);
      Interp2Patch(level_prev.imgf, x0, y0, &patch1_dt);
patch1_dt -= patch1_dt.sum() / patch1_dt.size();
      for (int loop = 0; loop < num_loop; ++loop) {
        if (!(rad <= x && rad <= y && x < x_max && y < y_max)) {
          x = -1.f, y = -1.f;  // Set invalid coordinates.
          break;
        }
        double e[2] = { 0, 0 }, Z[3] = { 0, 0, 0 };
        score = 0.0;

//        LOG(INFO) << "interp2 begin";
        Interp2Patch(level_cur.imgx, x, y, &patch0);
        patch_dx = (patch0 + patch1_dx) / 2;
        Interp2Patch(level_cur.imgy, x, y, &patch0);
        patch_dy = (patch0 + patch1_dy) / 2;
        Interp2Patch(level_cur.imgf, x, y, &patch0);
patch0 -= patch0.sum() / patch0.size();
        patch_dt = patch0 - patch1_dt;
        Z[0] -= patch_dx.square().sum();
        Z[1] -= (patch_dx * patch_dy).sum();
        Z[2] -= patch_dy.square().sum();
        e[0] += (patch_dx * patch_dt).sum();
        e[1] += (patch_dy * patch_dt).sum();
        score += patch_dt.abs().sum();
//        LOG(INFO) << "interp2 done " << Z[0] << ", " << Z[1] << ", " << Z[2]
//            << ", " << e[0] << ", " << e[1];
/*
        double mp = 0.0, mq = 0.0;
        for (int v = -rad; v <= rad; ++v) {
          for (int u = -rad; u <= rad; ++u) {
//            const double dx = Interp2(level_cur.imgx, x + u, y + v);
//            const double dy = Interp2(level_cur.imgy, x + u, y + v);
            const double dx = 0.5 * (Interp2(level_cur.imgx, x + u, y + v) +
                                     Interp2(level_prev.imgx, x0 + u, y0 + v));
            const double dy = 0.5 * (Interp2(level_cur.imgy, x + u, y + v) +
                                     Interp2(level_prev.imgy, x0 + u, y0 + v));
            const double p = Interp2(level_cur.imgf, x + u, y + v);
            const double q = Interp2(level_prev.imgf, x0 + u, y0 + v);
            const double dt = p - q;
//            const double dt = Interp2(level_cur.imgf, x + u, y + v) -
//                              Interp2(level_prev.imgf, x0 + u, y0 + v);
//            const double w = gaussian_kernel(u + rad, 0) *
//                             gaussian_kernel(v + rad, 0);
            const double w = 1.0;
            Z[0] -= w * dx * dx;
            Z[1] -= w * dx * dy;
            Z[2] -= w * dy * dy; 
            e[0] += w * dx * dt;
            e[1] += w * dy * dt;
            score += fabs(dt);
            mp += w * p;
            mq += w * q;
          }
        }
//        e[0] += mp - mq;
//        e[1] += mp - mq;
*/
        const double coef = 1.0 / (Z[0] * Z[2] - Z[1] * Z[1] + 1e-9);
        const double Z_inv[3] = { coef * Z[2], -coef * Z[1], coef * Z[0] };
        const double dx = Z_inv[0] * e[0] + Z_inv[1] * e[1];
        const double dy = Z_inv[1] * e[0] + Z_inv[2] * e[1];
        x -= dx * (loop + 1) / num_loop;
        y -= dy * (loop + 1) / num_loop;
      }
      if (!(rad <= x && rad <= y && x < x_max && y < y_max)) {
        x = -1.f, y = -1.f;
        break;
      }
    }
    ft.pos << x, y;
    ft.score = score;
  }
  return true;
}

// KLTTracker Implementation.

KLTTracker::KLTTracker()
    : frame_no_(0), next_id_(0), detect_count_(3),
      pyramid_builder_(FLAGS_klt_num_levels, FLAGS_klt_sigma) {
}

bool KLTTracker::Setup(const ArrayXXu8& image) {
  if (pyramid_builder_.Build(image, &prev_) == false) return false;

  features_.clear();
  KLTDetectFeatures(prev_, pyramid_builder_.gaussian_1d_kernel(),
                    FLAGS_klt_max_features, FLAGS_klt_min_cornerness,
                    &features_);
  for (int i = 0; i < features_.size(); ++i) {
    features_[i].id = next_id_++;
  }
  //LOG(INFO) << features_.size() << " detected features";
  return true;
}

void KLTTracker::Cleanup() {
}

bool KLTTracker::Process(const ArrayXXu8& image) {
  ++frame_no_;
ProfileBegin("11.pyrmid", &pdb_);
  ImagePyramid cur;
  if (pyramid_builder_.Build(image, &cur) == false) return false;
ProfileEnd("11.pyrmid", &pdb_);

ProfileBegin("12.track", &pdb_);
  const ArrayF& gaussian_kernel = pyramid_builder_.gaussian_1d_kernel();
  // Track existing features.
  KLTTrackFeatures(prev_, cur, gaussian_kernel, FLAGS_klt_num_loop,
                   &features_);
  const int dup_filter_rad = gaussian_kernel.size(); // / 2;
  FilterDuplicateTracks(dup_filter_rad, &features_);
  //LOG(INFO) << features_.size() << " corners tracked (" << next_id_ << ")";
ProfileEnd("12.track", &pdb_);

  // Detect new features if # features are below min threshold.
  if (features_.size() < FLAGS_klt_redetect_thr) {
ProfileBegin("13.detect", &pdb_);
    const int idx0 = features_.size();
    KLTDetectFeatures(cur, gaussian_kernel, FLAGS_klt_max_features,
                      FLAGS_klt_min_cornerness, &features_);
    for (int i = idx0; i < features_.size(); ++i) {
      features_[i].id = next_id_++;
    }
    //LOG(INFO) << features_.size() << " corners detected";
ProfileEnd("13.detect", &pdb_);
  }

  prev_.Swap(&cur);
  return true;
}

bool KLTTracker::ProcessIMU(const ArrayXXu8& image, const Mat3 cur_rot) {
  ++frame_no_;
ProfileBegin("11.pyrmid", &pdb_);
  ImagePyramid cur;
  if (pyramid_builder_.Build(image, &cur) == false) return false;
ProfileEnd("11.pyrmid", &pdb_);

ProfileBegin("12.track", &pdb_);
  const ArrayF& gaussian_kernel = pyramid_builder_.gaussian_1d_kernel();
  // Track existing features.
  KLTTrackFeaturesIMU(prev_, cur, gaussian_kernel, FLAGS_klt_num_loop, cur_rot,
                   &features_);
  const int dup_filter_rad = gaussian_kernel.size(); // / 2;
  FilterDuplicateTracksIMU(dup_filter_rad, &features_);
  //LOG(INFO) << features_.size() << " corners tracked (" << next_id_ << ")";
ProfileEnd("12.track", &pdb_);

  // Detect new features if # features are below min threshold.
  if (features_.size() < FLAGS_klt_redetect_thr) {
ProfileBegin("13.detect", &pdb_);
    const int idx0 = features_.size();
    KLTDetectFeatures(cur, gaussian_kernel, FLAGS_klt_max_features,
                      FLAGS_klt_min_cornerness, &features_);
    for (int i = idx0; i < features_.size(); ++i) {
      features_[i].id = next_id_++;
    }
    //LOG(INFO) << features_.size() << " corners detected";
ProfileEnd("13.detect", &pdb_);
  }

  prev_.Swap(&cur);
  return true;
}

// StereoKLTTracker Implementation.

StereoKLTTracker::StereoKLTTracker() : left_(), right_(), matches_() {
}

bool StereoKLTTracker::Setup(const ArrayXXu8& left_image,
                             const ArrayXXu8& right_image) {
  if (!right_.Setup(right_image) || !left_.Setup(left_image)) return false;
//  MatchFeaturesTwoWay(left_, right_, &matches_);
  MatchFeatures(left_, right_, 1, &matches_);
  LOG(INFO) << matches_.size() << " matches found"; // in " <<
//      left_to_right.size() << " and " << right_to_left.size();
  return true;
}

void StereoKLTTracker::Cleanup() {
  left_.Cleanup();
  right_.Cleanup();
}

bool StereoKLTTracker::Process(const ArrayXXu8& left_image,
                               const ArrayXXu8& right_image) {
  if (!right_.Process(right_image) || !left_.Process(left_image)) return false;
//  MatchFeaturesTwoWay(left_, right_, &matches_);
  MatchFeatures(left_, right_, 1, &matches_);
  LOG(INFO) << matches_.size() << " matches found"; // in " <<
//      left_to_right.size() << " and " << right_to_left.size();
  return true;
}

void StereoKLTTracker::MatchFeaturesTwoWay(const KLTTracker& left,
                                           const KLTTracker& right,
                                           vector<IDPair>* matches) {
  vector<IDPair> left_to_right, right_to_left;
  MatchFeatures(left, right, 1, &left_to_right);
  MatchFeatures(right, left, -1, &right_to_left);
  matches->clear();
  vector<IDPair>::const_iterator lb, ub,
      rl_begin = right_to_left.begin(), rl_end = right_to_left.end();
  for (int i = 0; i < left_to_right.size(); ++i) {
    const IDPair lr_pair = left_to_right[i];
    const int r_id = left_to_right[i].second;
    lb = lower_bound(rl_begin, rl_end, make_pair(lr_pair.second, 0));
    ub = lower_bound(rl_begin, rl_end, make_pair(lr_pair.second + 1, 0));
    if (lb != ub && lr_pair.first == lb->second) matches->push_back(lr_pair);
  }
  LOG(INFO) << matches->size() << " matches found in " <<
      left_to_right.size() << " and " << right_to_left.size();
}

void StereoKLTTracker::MatchFeatures(const KLTTracker& ref,
                                     const KLTTracker& dst,
                                     int match_dir,  // either +1 or -1.
                                     vector<IDPair>* matches) {
  const vector<KLTFeat>& ref_feats = ref.features();
  const vector<KLTFeat>& dst_feats = dst.features();
  const ImagePyramid& ref_pyr = ref.image_pyramid();
  const ImagePyramid& dst_pyr = dst.image_pyramid();
  const double min_match_score = -pow(FLAGS_klt_stereo_min_match_score, 2);
  const int max_disp = FLAGS_klt_stereo_max_disp;
  const double sig = FLAGS_klt_sigma;
  const int y_off = FLAGS_klt_stereo_y_offset;
  const int win = 2 * ceil(3 * sig) + 1;
  ImagePyramid::ArrayF ref_patch(win, win), dst_patch(win, win);
  matches->clear();
  for (int i = 0; i < ref_feats.size(); ++i) {
    const KLTFeat& ref_ft = ref_feats[i];
    const int lvl = ref_ft.level;
    const double lc = 1.0 / (1L << lvl);
    const float rx = ref_ft.pos(0) * lc, ry = ref_ft.pos(1) * lc;
    int match_idx = -1;
    double match_score = min_match_score;
    Interp2Patch(ref_pyr.level(lvl).imgx, rx, ry, &ref_patch);
    for (int j = 0; j < dst_feats.size(); ++j) {
      const KLTFeat& dst_ft = dst_feats[j];
      if (lvl != dst_ft.level) continue;
      const float dx = dst_ft.pos(0) * lc, dy = dst_ft.pos(1) * lc;
      const float disp = match_dir * (rx - dx);  // Disparty.
      if (fabs(ry - dy) > y_off || disp < -2 || disp > max_disp) continue;
      Interp2Patch(dst_pyr.level(lvl).imgx, dx, dy, &dst_patch);
      const double score = -(ref_patch - dst_patch).square().mean();
//      LOG_FIRST_N(INFO, 3) << endl << ref_patch;
//      LOG_FIRST_N(INFO, 3) << endl << dst_patch;
//      LOG(INFO) << i << "," << j << ": " << lvl << " (" << rx << "," << ry
//          << "), (" << dx << "," << dy << ") : " << score;
      if (score > match_score) {
        match_score = score;
        match_idx = j;
      }
    }
    if (match_idx >= 0) matches->push_back(IDPair(i, match_idx));
  }
//  LOG(INFO) << matches->size() << " matches found.";
}

}  // namespace rvslam

