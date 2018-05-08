// image_file.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _RVSLAM_IMAGE_FILE_H_
#define _RVSLAM_IMAGE_FILE_H_

#include <stdarg.h>
#include <stdio.h>
#include <cstdlib>
#include <string>
#include "image.h"

namespace rvslam {

inline std::string StringPrintf(const char* fmt, ...) {
  char buf[1024];
  va_list a;
  va_start(a, fmt);
  vsprintf(buf, fmt, a);
  va_end(a);
  return std::string(buf);
}

bool ReadImageGray8(const std::string& filepath, ImageGray8* image);
bool ReadImageGray8(const std::string& filepath, MCImageGray8* image);
bool ReadImageGray16(const std::string& filepath, ImageGray16* image);
bool ReadImageGray16(const std::string& filepath, MCImageGray16* image);
bool ReadImageRGB8(const std::string& filepath, MCImageRGB8* image);

bool WriteImageGray8(const ImageGray8& image, const std::string& filepath);
bool WriteImageGray8(const MCImageGray8& image, const std::string& filepath);
bool WriteImageRGB8(const MCImageRGB8& image, const std::string& filepath);

template <class T>
bool ReadImage(const std::string& filepath, Image<T>* image);

template <class T, int D>
bool ReadImage(const std::string& filepath, MCImage<T, D>* image);

template <class T>
bool WriteImage(const Image<T>& image, const std::string& filepath);

template <class T, int D>
bool WriteImage(const MCImage<T, D>& image, const std::string& filepath);

}  // namespace rvslam
#endif  // _RVSLAM_IMAGE_FILE_H_
