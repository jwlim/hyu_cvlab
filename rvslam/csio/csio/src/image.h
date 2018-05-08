// image.h
//
//  Image using Eigen Matrix class.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _CSIO_IMAGE_H_
#define _CSIO_IMAGE_H_

#include <iostream>
#include <map>
#include <string>

#include <Eigen/Dense>

namespace csio {

template<typename _Scalar, int DEPTH, int SIZE = Eigen::Dynamic>
class Image : public Eigen::Array<_Scalar, DEPTH, SIZE> {
 public:
  Image() : Eigen::Array(DEPTH, SIZE), width_(0), height_(0) {}
  Image(int width, int heght)
      : Eigen::Array(DEPTH, SIZE), width_(width), height_(height) {
    // ASSERT(width * height > SIZE)
  }

  typedef _Scalar Scalar;
  typedef Eigen::Array<_Scalar, Eigen::Dynamic, Eigen::Dynamic> ArrayType;

  const int depth() const { return DEPTH; }
  const int width() const { return width_; }
  const int height() const { return height_; }
  const int size() const { return width_ * height_; }
  const int num_pixels() const { return width_ * height_; }

  void Resize(int width, int height) { resize(width, height); }
  void resize(int width, int height) {
    Eigen::Array::resize(DEPTH, (width_ = width) * (height_ = height));
  }

  Eigen::Stride<DEPTH, Eigen::Dynamic> stride() const {
    return Eigen::Stride<DEPTH, Eigen::Dynamic>(DEPTH, width_ * height_);
  }

  Image GetSubimage(int x, int y, int w, int h) const {
    ArrayType ret = Map(row(ch).data(), width_, height_, stride());
    return ret;
  }

  ArrayType ToArray(int ch = 0) const {
    Eigen::Stride<DEPTH, Eigen::Dynamic> stride(DEPTH, width_ * height_);
    ArrayType ret = Map(row(ch).data(), width_, height_, stride);
    return ret;
  }

 private:
  int width_, height_;
};
/*
inline std::string ParseTypeStr(const std::string& type_str,
                                std::map<std::string, std::string>* cfg) {
  std::string str = type_str;
  std::string type = StrTok(" \t\r\n", &str);
  if (!str.empty() && !cfg) SplitStr(str, " \t\r\n", "'\"", "=", cfg);
  return type;
}

inline std::string MakeTypeStr(const std::string& type,
                               const std::map<std::string, std::string>& cfg) {
  std::string str = type;
  for (std::map<std::string, std::string>::const_iterator it = cfg.begin();
       it != cfg.end(); ++it) {
    std::string value = it->second;
    if (Contains(value, " \t\r\n")) value = "'" + value + "'";
    str += ";" + it->first + "=" + value;
  }
  return str;
}
*/
}  // namespace csio
#endif  // _CSIO_IMAGE_H_

