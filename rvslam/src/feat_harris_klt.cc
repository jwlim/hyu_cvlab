// feat_harris_klt.cc
//
// Author : Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include "feat_harris_klt.h"

#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "image_util.h"  // Convolve.

#include "rvslam_profile.h"
extern rvslam::ProfileDBType pdb_;

using namespace std;

DEFINE_int32(klt_num_levels, 3,
             "The number of image pyramid levels in KLT Tracker.");
DEFINE_double(klt_sigma, 1.5, "Gaussian sigma for KLT tracker.");
DEFINE_double(klt_min_cornerness, 1.5, "The mininum cornerness response.");
DEFINE_int32(klt_max_features, 3000,
             "The maximum number of features to track.");
DEFINE_int32(klt_redetect_thr, 600, "The threshold to redetect features.");
DEFINE_int32(klt_num_loop, 10, "The number of loops in each iteration.");
DEFINE_int32(klt_detect_level, 0, "The pyramid level to detect features.");

namespace {

typedef Eigen::ArrayXXf ArrayF;
using namespace rvslam;

template <class T> inline
string ToString(const T& v) {
  stringstream ss;
  ss << v;
  return ss.str();
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
                 vector<HarrisCorner>* corners) {
  // Compute cornerness.
  const int rad = gaussian_kernel.rows() / 10;
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
  // Mark existing features.
  const double kExistingFeature = 1e9;
  const double level_scale = (1L << level_idx);
  for (int i = 0; i < corners->size(); ++i) {
    HarrisCorner& ft = corners->at(i);
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
        corners->push_back(HarrisCorner(-1, level_idx, pos, cornerness_x_y));
      }
    }
  }
}

void SortAndFilterCorners(int num_features, int max_features,
                          vector<HarrisCorner>* corners) {
  stable_sort(corners->begin() + num_features, corners->end());
  if (corners->size() > max_features) corners->resize(max_features);
}

size_t FilterDuplicateTracks(int rad, vector<HarrisCorner>* corners) {
  multimap<float, int> ypos;
  for (int i = 0; i < corners->size(); ++i) {
    const Eigen::Vector2f& p = corners->at(i).pos;
    if (p[0] >= 0 && p[1] >= 0) ypos.insert(make_pair(p[1], i));
  }
  vector<HarrisCorner> tmp_corners;
  tmp_corners.reserve(corners->size());
  for (int i = 0; i < corners->size(); ++i) {
    const HarrisCorner& ft = corners->at(i);
    const int ft_x = ft.pos[0], ft_y = ft.pos[1];
    if (ft_x < 0 || ft_y < 0) continue;
    multimap<float, int>::const_iterator it, it_lb, it_ub;
    it_lb = ypos.lower_bound(ft_y - rad);
    it_ub = ypos.upper_bound(ft_y + rad);
    bool duplicate = false;
    for (it = it_lb; it != it_ub && !duplicate; ++it) {
      const HarrisCorner& ft2 = corners->at(it->second);
      duplicate = (ft_x - rad < ft2.pos[0] && ft2.pos[0] < ft_x + rad &&
                   ft.score < ft2.score);
    }
    if (!duplicate) {
      tmp_corners.push_back(ft);
    }
  }
  corners->swap(tmp_corners);
  return corners->size();
}

bool DetectHarrisCorners(const ImagePyramid& pyr, const ArrayF& gaussian_kernel,
                       int max_features, double min_cornerness,
                       vector<HarrisCorner>* corners) {
  const int org_size = corners->size();
  const int rad = gaussian_kernel.rows() / 2;
  for (int l = 0; corners->size() < max_features && l < pyr.num_levels(); ++l) {
    const ImagePyramid::Level& level = pyr[l];
    FindCorners(level, l, gaussian_kernel, min_cornerness, corners);
    SortAndFilterCorners(org_size, max_features, corners);
  }
  for (int i = org_size; i < corners->size(); ++i) {
    HarrisCorner& ft = corners->at(i);
    ft.score = 0.0;
  }
  return true;
}

bool KLTTrackFeatures(const ImagePyramid& prev, const ImagePyramid& cur,
                      const ArrayF& gaussian_kernel, const int num_loop,
                      vector<HarrisCorner>* corners) {
  if (corners->size() <= 0) return true;
  // Track features from the top level to the bottom.
  const int rad = gaussian_kernel.rows() / 2;
  const int win_size = (2 * rad + 1);
  ArrayF patch_dx(win_size, win_size), patch_dy(win_size, win_size);
  ArrayF patch_dt(win_size, win_size);
  ArrayF patch0(win_size, win_size);
  ArrayF patch1_dx(win_size, win_size);
  ArrayF patch1_dy(win_size, win_size);
  ArrayF patch1_dt(win_size, win_size);
  for (int i = 0; i < corners->size(); ++i) {
    const int num_levels = cur.num_levels();
    const double pow2level = (1L << num_levels);
    HarrisCorner& ft = corners->at(i);
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
    ft.pos << x, y;
    ft.score = score;
  }
  return true;
}

}  // anonymous namespace

namespace rvslam {

inline bool operator<(const HarrisCorner& ft1, const HarrisCorner& ft2) {
  return ft1.score < ft2.score;
}

//-----------------------------------------------------------------------------

HarrisCornerDetector::HarrisCornerDetector()
    : pyr_(), pyramid_builder_(FLAGS_klt_num_levels, FLAGS_klt_sigma) {
}

HarrisCornerDetector::~HarrisCornerDetector() {
  Cleanup();
}

bool HarrisCornerDetector::SetImage(const ImageType& image) {
ProfileBegin("11.pyrmid", &pdb_);
  if (pyramid_builder_.Build(image, &pyr_) == false) return false;
ProfileEnd("11.pyrmid", &pdb_);
  return true;
}

bool HarrisCornerDetector::Detect(FeatLocMatType* ftlocs) {
  std::vector<HarrisCorner> corners;
  corners.reserve(FLAGS_klt_max_features);
  for (int i = 0; i < ftlocs->cols(); ++i) {
    corners.push_back(HarrisCorner(0, 0, ftlocs->col(i), 0.0));
  }
  DetectHarrisCorners(pyr_, pyramid_builder_.gaussian_1d_kernel(),
                      FLAGS_klt_max_features, FLAGS_klt_min_cornerness,
                      &corners);
  LOG(INFO) << corners.size() << " detected features";
  return true;
}

bool HarrisCornerDetector::GetDescriptors(const FeatLocMatType& ftlocs,
                                          FeatDescMatType* ftdescs) {
  return false;
}

void HarrisCornerDetector::Cleanup() {
}

//-----------------------------------------------------------------------------

KLTTracker::KLTTracker() {
}

KLTTracker::~KLTTracker() {
}

bool KLTTracker::Process(const ImageType& image) {
  return false;
}

}  // namespace rvslam
