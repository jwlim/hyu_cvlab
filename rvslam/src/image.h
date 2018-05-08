// image.h
//
//  MCImage using Eigen Matrix class.
//  Multi-channel image class is also defined.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _RVSLAM_IMAGE_H_
#define _RVSLAM_IMAGE_H_

#include <iostream>
#include <map>
#include <string>
#include <stdint.h>
#include <stdarg.h>
#include <Eigen/Dense>

namespace rvslam {

// In rvslam, grayscale images are Eigen Array in row-major order.
//  (0,0) +------------------+ (w-1,0)
//        |                  |
//        |                  |
//        |                  |
//        |                  |
//        +------------------+ (w-1,h-1)
// The current implementation only adds the width() and height() member
// functions.

template<typename SC, int WIDTH = Eigen::Dynamic, int HEIGHT = Eigen::Dynamic>
class Image : public Eigen::Array<SC, WIDTH, HEIGHT> {
 public:
  typedef SC Scalar;
  typedef Scalar PixelType;
  typedef Eigen::Array<Scalar, WIDTH, HEIGHT> BaseType;
  typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> ArrayType;

  Image() : BaseType() {}
  Image(int width, int height) : BaseType(width, height) {}
  template <int W, int H>
  Image(const Eigen::Array<Scalar, W, H>& image) : BaseType(image) {}

  const int depth() const { return 1; }
  const int width() const { return this->rows(); }
  const int height() const { return this->cols(); }
  const int size() const { return width() * height(); }
  const int num_channels() const { return depth(); }
  const int num_pixels() const { return size(); }

  Scalar& At(int x, int y) { return BaseType::operator()(x, y); }
  const Scalar& At(int x, int y) const { return BaseType::operator()(x, y); }

  Scalar& At(int i) { return At(i % width(), i / width()); }
  const Scalar& At(int i) const { return At(i % width(), i / width()); }
  Scalar& operator[](int i) { return At(i); }
  const Scalar& operator[](int i) const { return At(i); }

  void Swap(Image* image) { BaseType::swap(*image); }
  void Resize(int width, int height) { BaseType::resize(width, height); }

  template <typename S>
  Eigen::Array<S, WIDTH, HEIGHT> base_cast() const {
    return BaseType::template cast<S>();
  }
  template <typename S>
  Image<S, WIDTH, HEIGHT> cast() const {
    Image<S, WIDTH, HEIGHT> ret;
    ret.FromArray(this->template base_cast<S>(), width(), height());
    return ret;
  }

  template <int W, int H>
  Image& operator=(const Eigen::Array<Scalar, W, H>& image) {
    BaseType::operator=(image);
    return *this;
  }

  BaseType ToArray() const {
    BaseType ret = BaseType::Map(this->data(), 1, num_pixels());
    return ret;
  }

  void FromArray(const ArrayType& array, int width, int height) {
    BaseType::operator=(BaseType::Map(
        reinterpret_cast<const Scalar*>(array.data()), width, height));
  }

  void FromBuffer(const void* buf, int width, int height) {
    BaseType::operator=(BaseType::Map(
        reinterpret_cast<const Scalar*>(buf), width, height));
  }
};

typedef Image<uint8_t> ImageGray8;
typedef Image<uint16_t> ImageGray16;
typedef Image<uint32_t> ImageGray32;
typedef Image<short> ImageShort;
typedef Image<long> ImageLong;
typedef Image<float> ImageFloat;
typedef Image<double> ImageDouble;

// To represent color images or other multi-channel images, the following
// representation using Eigen Array is used.
//  (0,0) +------------------+
//  ch:0  |                  |-+
//        |                  | |-+
//        |                  | | |
//        |                  | | |
//        +------------------+ | |
//          +------------------+ |
//            +------------------+ (w-1,h-1), ch:2
//  ==>     +-+-+-+-----...--+-+-+-+------------------------------...-----+
//   ch:0 > | | | |          | | | |                                      |
//   ch:1 > | | | |          | | | |                                      |
//   ch:2 > | | | |          | | | |                                      |
//          +-+-+-+-----...--+-+-+-+------------------------------...-----+
//           : : ^(2,0)       : : ^(1,1)
//           : (1,0)          : (0,1)
//           (0,0)            (w-1,0)

template<typename _Scalar, int DEPTH, int SIZE = Eigen::Dynamic>
class MCImage : public Eigen::Array<_Scalar, DEPTH, SIZE> {
 public:
  typedef _Scalar Scalar;
  typedef Eigen::Array<Scalar, DEPTH, SIZE> BaseType;
  typedef Eigen::Array<Scalar, DEPTH, 1> PixelType;
  typedef typename BaseType::ColXpr PixelRetType;
  typedef typename BaseType::ConstColXpr PixelConstRetType;
  typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> ArrayType;

  MCImage() : BaseType(), width_(0), height_(0) {}
  MCImage(int width, int height, int depth = 1)
      : BaseType(depth, width * height), width_(width), height_(height) {}
  template <int D, int S>
  MCImage(const MCImage<Scalar, D, S>& image)
      : BaseType(image), width_(image.width()), height_(image.height()) {}

  const int depth() const { return this->rows(); }
  const int width() const { return width_; }
  const int height() const { return height_; }
  const int size() const { return width_ * height_; }
  const int num_channels() const { return this->rows(); }
  const int num_pixels() const { return width_ * height_; }
  const int colidx(int x, int y) const { return x + y * width_; }

  PixelRetType At(int x, int y) { return this->col(colidx(x, y)); }
  PixelConstRetType At(int x, int y) const { return this->col(colidx(x, y)); }
  PixelRetType operator()(int x, int y) { return At(x, y); }
  PixelConstRetType operator()(int x, int y) const { return At(x, y); }

  PixelRetType& At(int i) { return this->col(i); }
  PixelConstRetType& At(int i) const { return this->col(i); }
  PixelRetType& operator[](int i) { return At(i); }
  PixelConstRetType& operator[](int i) const { return At(i); }

  Scalar& At(int x, int y, int d) { return this->coeffRef(d, colidx(x, y)); }
  const Scalar& At(int x, int y, int d) const {
    return this->coeff(d, colidx(x, y));
  }
  Scalar& operator()(int x, int y, int d) { return At(x, y, d); }
  const Scalar& operator()(int x, int y, int d) const { return At(x, y, d); }

  Eigen::Stride<Eigen::Dynamic, DEPTH> array_stride() const {
    return Eigen::Stride<Eigen::Dynamic, DEPTH>(depth() * width_, depth());
  }

  void Swap(MCImage* image) { swap(image); }
  void Resize(int width, int height) { resize(width, height); }
  void Resize(int width, int height, int depth) {
    resize(width, height, depth);
  }

  MCImage& operator=(const MCImage& image) {
    BaseType::operator=(image);
    width_ = image.width_, height_ = image.height_;
    return *this;
  }

  template <typename S>
  Eigen::Array<S, DEPTH, SIZE> base_cast() const {
    return BaseType::template cast<S>();
  }
  template <typename S>
  MCImage<S, DEPTH, SIZE> cast() const {
    MCImage<S, DEPTH, SIZE> ret;
    ret.FromArray(this->template base_cast<S>(), width_, height_);
    return ret;
  }

  // TODO: Implement GetSubimage().
  // MCImage GetSubimage(int x, int y, int w, int h) const;

  template <typename T, int D, int S>
  MCImage& Copy(const MCImage<T, D, S>& image) {
    this->resize(image.width(), image.height(), image.depth());
    BaseType::operator=(image.template base_cast<Scalar>());
    return *this;
  }

  ArrayType GetPlane(int ch = 0) const {
    return ArrayType::Map(this->data() + ch, width_, height_, array_stride());
  }

  void SetPlane(int ch, const ArrayType& gray_image) {
    this->resize(gray_image.rows(), gray_image.cols(), depth());
    ArrayType::Map(this->data() + ch, width_, height_, array_stride()) =
        gray_image;
  }

  void SetAllPlanes(const ArrayType& gray_image) {
    const int w = gray_image.rows(), h = gray_image.cols();
    ArrayType row = ArrayType::Map(gray_image.data(), 1, w * h);
    FromArray(row.replicate(depth(), 1), w, h);
  }

  BaseType ToArray() const { return *this; }

  void FromArray(const ArrayType& array, int width, int height) {
    BaseType::operator=(array);
    width_ = width, height_ = height;
  }

  void FromBuffer(const void* buf, int w, int h, int d) {
    this->resize(w, h, d);
    BaseType::operator=(BaseType::Map(reinterpret_cast<const Scalar*>(buf),
                                      depth(), size()));
  }

  template <typename T>
  void Collapse(Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>* ret) const {
    Eigen::Array<T, DEPTH, SIZE> casted = this->template base_cast<T>();
    Eigen::Array<T, 1, SIZE> avg = casted.colwise().mean();
    *ret = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> >(
        avg.data(), width_, height_);
  }

  static PixelType MakePixel(const Scalar& v) {
    return (PixelType() << v).finished();
  }
  static PixelType MakePixel(const Scalar& v0, const Scalar& v1) {
    return (PixelType() << v0, v1).finished();
  }
  static PixelType MakePixel(const Scalar& v0, const Scalar& v1,
                             const Scalar& v2) {
    return (PixelType() << v0, v1, v2).finished();
  }
  static PixelType MakePixel(const Scalar& v0, const Scalar& v1,
                             const Scalar& v2, const Scalar& v3) {
    return (PixelType() << v0, v1, v2, v3).finished();
  }

 private:
  // Prevent calling BaseType's resize and swap function. Use Resize and Swap.
  void resize(int width, int height, int depth = DEPTH) {
    BaseType::resize(depth, (width_ = width) * (height_ = height));
  }
  void swap(MCImage* image) {
    std::swap(width_, image->width_);
    std::swap(height_, image->height_);
    BaseType::swap(*image);
  }

  // Image width and height.
  int width_, height_;
};

typedef MCImage<uint8_t, 1> MCImageGray8;
typedef MCImage<uint16_t, 1> MCImageGray16;
typedef MCImage<uint8_t, 3> MCImageRGB8;
typedef MCImage<short, 1> MCImageShort;
typedef MCImage<float, 1> MCImageFloat;
typedef MCImage<uint8_t, Eigen::Dynamic> MCImageU8;
typedef MCImage<uint16_t, Eigen::Dynamic> MCImageU16;
typedef MCImage<uint32_t, Eigen::Dynamic> MCImageU32;

inline MCImageRGB8::PixelType MakePixelRGB8(uint8_t r, uint8_t g, uint8_t b) {
  return MCImageRGB8::MakePixel(r, g, b);
}

template <typename T>
inline void RGB8ToGray(const MCImageRGB8& rgb, Image<T>* gray) {
  Eigen::Array<T, 1, Eigen::Dynamic> avg =
      rgb.base_cast<float>().colwise().mean().cast<T>();
  gray->FromBuffer(avg.data(), rgb.width(), rgb.height());
}

inline void RGB8ToGray8(const MCImageRGB8& rgb, ImageGray8* gray) {
  Eigen::Array<uint8_t, 1, Eigen::Dynamic> avg =
      rgb.base_cast<float>().colwise().mean().cast<uint8_t>();
  gray->FromBuffer(avg.data(), rgb.width(), rgb.height());
}

inline void RGB8ToGray8(const MCImageRGB8& rgb, MCImageGray8* gray) {
  Eigen::Array<uint8_t, 1, Eigen::Dynamic> avg =
      rgb.base_cast<float>().colwise().mean().cast<uint8_t>();
  gray->FromBuffer(avg.data(), rgb.width(), rgb.height(), 1);
}

inline void Gray8ToRGB8(const ImageGray8& gray, MCImageRGB8* rgb) {
  Eigen::Array<uint8_t, 3, Eigen::Dynamic> rep =
      gray.ToArray().replicate<3, 1>();
  rgb->FromBuffer(rep.data(), gray.width(), gray.height(), 3);
}

inline void Gray8ToRGB8(const MCImageGray8& gray, MCImageRGB8* rgb) {
  Eigen::Array<uint8_t, 3, Eigen::Dynamic> rep = gray.replicate<3, 1>();
  rgb->FromBuffer(rep.data(), gray.width(), gray.height(), 3);
}

}  // namespace rvslam
#endif  // _RVSLAM_IMAGE_H_

