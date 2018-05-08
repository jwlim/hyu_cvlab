// image_file.h
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _CSIO_IMAGE_FILE_H_
#define _CSIO_IMAGE_FILE_H_

#include <string>

namespace csio {

namespace pix {
enum {
  UNKNOWN = 0, GRAY8 = 1, GRAY16 = 2, RGB8 = 3
};
}  // namespace pix

bool ReadImageFromPNG(const std::string& filepath,
                      int* pixf, int* width, int* height, std::string* data);

bool WriteImageToPNG(const void* data, int pixf, int width, int height,
                     int pitch, const std::string& filepath);

inline bool WriteGRAY8ToPNG(const void* data, int width, int height, int pitch,
                            const std::string& filepath) {
  return WriteImageToPNG(data, pix::GRAY8, width, height, pitch, filepath);
}

inline bool WriteGRAY16ToPNG(const void* data, int width, int height, int pitch,
                             const std::string& filepath) {
  return WriteImageToPNG(data, pix::GRAY16, width, height, pitch, filepath);
}

inline bool WriteRGB8ToPNG(const void* data, int width, int height, int pitch,
                           const std::string& filepath) {
  return WriteImageToPNG(data, pix::RGB8, width, height, pitch, filepath);
}

}  // namespace csio
#endif  // _CSIO_IMAGE_FILE_H_

