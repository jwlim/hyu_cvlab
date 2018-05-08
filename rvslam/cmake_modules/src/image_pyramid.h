// image_pyramid.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _RVSLAM_IMAGE_PYRAMID_H_
#define _RVSLAM_IMAGE_PYRAMID_H_

#include <cmath>
#include <vector>
#include <glog/logging.h>
#include <Eigen/Dense>
#include "image_util.h"  // BuildGaussian1D, Convolve.

namespace rvslam {

class ImagePyramid {
 public:
  typedef Eigen::ArrayXXf ArrayF;
  struct Level {
    ArrayF imgf, imgx, imgy;
  };

  ImagePyramid() : levels_() {}
  ImagePyramid(const ImagePyramid& p) : levels_(p.levels_) {}
  ~ImagePyramid() {}

  void Copy(const ImagePyramid& p) { levels_ = p.levels_; }
  void Swap(ImagePyramid* p) { levels_.swap(p->levels_); }

  size_t num_levels() const { return levels_.size(); }
  const Level& level(size_t l) const { return levels_[l]; }
  const Level& operator[](size_t l) const { return levels_[l]; }

  template <typename Pix> inline
  bool Build(const Eigen::ArrayBase<Pix>& image, size_t num_levels,
             const ArrayF& gaussian_1d, const ArrayF& diff_row);

 private:
  std::vector<Level> levels_;
  ArrayF gaussian_1d_, dx_;
};

class ImagePyramidBuilder {
 public:
  typedef Eigen::ArrayXXf ArrayF;

  inline ImagePyramidBuilder(size_t num_levels, double gaussian_sigma,
                             size_t kernel_radius = 0);
  ~ImagePyramidBuilder() {}

  template <typename Pix>
  bool Build(const Eigen::ArrayBase<Pix>& image, ImagePyramid* pyr) const {
    return pyr->Build(image, num_levels_, gaussian_1d_, diff_row_);
  }

  const ArrayF& gaussian_1d_kernel() const { return gaussian_1d_; }

 private:
  size_t num_levels_;
  double gaussian_sigma_;
  ArrayF gaussian_1d_, diff_row_;
};

template <typename Pix>
bool ImagePyramid::Build(const Eigen::ArrayBase<Pix>& image, size_t num_levels,
                         const ArrayF& gaussian_1d, const ArrayF& diff_row) {
  if (num_levels <= 0) {
    LOG(ERROR) << "Invalid num_levels (" << num_levels << ").";
    return false;
  }
  levels_.resize(num_levels);
  int num_rows = image.rows(), num_cols = image.cols();
  // Build the levels of the pyramid from the lowest level.
  levels_[0].imgf = image.template cast<float>();
  levels_[0].imgx.resize(num_rows, num_cols);
  levels_[0].imgy.resize(num_rows, num_cols);
  for (int i = 1; i < levels_.size(); ++i) {
    num_rows /= 2, num_cols /= 2;
    levels_[i].imgf.resize(num_rows, num_cols);
    levels_[i].imgx.resize(num_rows, num_cols);
    levels_[i].imgy.resize(num_rows, num_cols);
    // Resize the upper level image into half.
    const ArrayF& prev_imgf = levels_[i - 1].imgf;
    ArrayF& cur_imgf = levels_[i].imgf;
    for (int r = 0, r0 = 0; r < num_rows; ++r, r0 += 2) {
      for (int c = 0, c0 = 0; c < num_cols; ++c, c0 += 2) {
        cur_imgf(r, c) =
            0.25f * (prev_imgf(r0, c0) + prev_imgf(r0, c0 + 1) +
                     prev_imgf(r0 + 1, c0) + prev_imgf(r0 + 1, c0 + 1));
      }
    }
  }
  // Blur the imgf and build imgx and imgy at each level.
  for (int i = 0; i < levels_.size(); ++i) {
    Level& lvl = levels_[i];
    const ArrayF& imgf = lvl.imgf;
    Convolve(lvl.imgf, gaussian_1d, &lvl.imgx);
    Convolve(lvl.imgx, gaussian_1d.transpose(), &lvl.imgf);
    Convolve(lvl.imgf, diff_row, &lvl.imgx);
    Convolve(lvl.imgf, diff_row.transpose(), &lvl.imgy);
  }
  return true;
}

ImagePyramidBuilder::ImagePyramidBuilder(
    size_t num_levels, double gaussian_sigma, size_t kernel_radius)
    : num_levels_(num_levels), gaussian_sigma_(gaussian_sigma) {
  if (kernel_radius <= 0) kernel_radius = ::ceil(3 * gaussian_sigma_);
  const int kernel_size = 2 * kernel_radius + 1;
  VLOG(2) << "ImagePyramidBuilder: " << gaussian_sigma_ << " " << kernel_size;
  gaussian_1d_.resize(std::max(3, kernel_size), 1);
  BuildGaussian1D(gaussian_sigma_, &gaussian_1d_);
  diff_row_.resize(3, 1);
  diff_row_ << -0.5, 0, 0.5;
}

}  // namespace rvslam
#endif  // _RVSLAM_IMAGE_PYRAMID_H_

