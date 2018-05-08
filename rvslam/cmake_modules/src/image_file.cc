// image_file.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#include "image_file.h"

#include <stdio.h>
#include <algorithm>
#include <cctype>
#include <vector>

#include <png.h>
#include <glog/logging.h>

using namespace std;

namespace rvslam {

namespace {

enum { GRAY8, GRAY16, RGB24 };

struct RawImageBuffer {
  string data;
  int pixfmt, width, height, pitch;
  RawImageBuffer() : data(), pixfmt(GRAY8), width(0), height(0), pitch(0) {}
};

string GetFileExtension(const string& filepath) {
  size_t dot_pos = filepath.rfind(".");
  if (dot_pos == string::npos) return string();
  string ext = filepath.substr(dot_pos);
  if (!ext.empty()) ext = ext.substr(1);
  transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

bool ReadRawImageBufferFromPNG(const string& filepath, RawImageBuffer* imgbuf) {
  // Open the image file and check if it's a valid png file.
  FILE* fp = fopen(filepath.c_str(), "rb");
  if (!fp) {
    LOG(ERROR) << "Failed to open file " << filepath;
    return false;
  }
  png_byte png_header[8];  // 8 is the maximum size that can be checked.
  fread(png_header, 1, 8, fp);
  if (png_sig_cmp(png_header, 0, 8)) {
    LOG(ERROR) << "Incorrect PNG header " << filepath;
    fclose(fp);
    return false;
  }
  // Initialize the png struct to load the content.
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                           NULL, NULL, NULL);
  if (!png) {
    LOG(ERROR) << "png_create_read_struct failed " << filepath;
    fclose(fp);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    LOG(ERROR) << "png_create_info_struct failed " << filepath;
    png_destroy_read_struct(&png, NULL, NULL);
    fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during setting up png read " << filepath;
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return false;
  }
  png_init_io(png, fp);
  png_set_sig_bytes(png, 8);
  png_read_info(png, info);

  const png_byte ctype = png_get_color_type(png, info);
  const png_byte bit_depth = png_get_bit_depth(png, info);
  if (ctype == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
  if ((ctype & PNG_COLOR_MASK_ALPHA)) png_set_strip_alpha(png);
  if (bit_depth < 8)  png_set_packing(png);

  imgbuf->pixfmt = GRAY8;
  if (ctype == PNG_COLOR_TYPE_RGB || ctype == PNG_COLOR_TYPE_RGB_ALPHA) {
    imgbuf->pixfmt = RGB24;
    if (bit_depth > 8)  png_set_strip_16(png);
  } else if (bit_depth > 8) {
    imgbuf->pixfmt = GRAY16;
    png_set_swap(png);  // LSB -> MSB
  } else if (ctype == PNG_COLOR_TYPE_PALETTE) {
    LOG(ERROR) << "palette ctype not supported.";
  }
  // png_set_interlace_handling(png);
  // png_read_update_info(png, info);

  imgbuf->width = png_get_image_width(png, info);
  imgbuf->height = png_get_image_height(png, info);
  imgbuf->pitch = imgbuf->width * (imgbuf->pixfmt == RGB24 ? 3 :
                                   imgbuf->pixfmt == GRAY16 ? 2 : 1);
  VLOG(2) << imgbuf->width << "x" << imgbuf->height << ", " << imgbuf->pitch;
  VLOG(2) << "ctype: " << (int) ctype << ", bit_depth: " << (int) bit_depth;
  VLOG(2) << "ctype: " << (int) PNG_COLOR_TYPE_RGB;
  VLOG(2) << "ctype: " << (int) PNG_COLOR_TYPE_RGB_ALPHA;
  imgbuf->data.resize(imgbuf->pitch * imgbuf->height);
  vector<png_bytep> rows(imgbuf->height);
  for (int i = 0; i < imgbuf->height; ++i) {
    rows[i] = (png_bytep) &imgbuf->data[i * imgbuf->pitch];
  }
  // Read image from the file.
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during reading a png image " << filepath;
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    return false;
  }
  png_read_image(png, &rows[0]);

  png_destroy_read_struct(&png, &info, NULL);
  fclose(fp);
  return true;
}

//----------------------------------------------------------------------------

bool WriteRawImageBufferToPNG(const void* data, int pixfmt,
                              int width, int height, int pitch,
                              const string& filepath) {
  if (width <= 0 || height <= 0) {
    LOG(ERROR) << "Empty image " << filepath;
    return false;
  }
//  ushort pixf = ib.pixf;
//  if (pix::is_bayer(pixf))
//    pixf = (pixf & pix::BAYER16_MASK)? pix::GRAY16 : pix::GRAY8;
//  if (pixf!=pix::GRAY8 && pixf!=pix::GRAY16 && pixf!=pix::RGB24)
//    HMSGRET(false, ("imwrite_png(%s): unsupported image format %d:'%s' (gray8,gray16,rgb24 only)", path, pixf, pix::pixf2str(pixf)));

  FILE *fp = fopen(filepath.c_str(), "wb");
  if (!fp) {
    LOG(ERROR) << "Failed to open " << filepath;
    return false;
  }
  // Initialize png structs.
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                            NULL, NULL, NULL);
  if (!png) {
    LOG(ERROR) << "png_create_write_struct failed " << filepath;
    fclose(fp);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    LOG(ERROR) << "png_create_info_struct failed " << filepath;
    png_destroy_write_struct(&png, NULL);
    fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during setting up png write " << filepath;
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_init_io(png, fp);

  // Write png header.
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during writting png header " << filepath;
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_set_compression_level(png, 3);
  png_set_compression_mem_level(png, 9);
  png_set_IHDR(
      png, info, width, height,
      pixfmt == GRAY16 ? 16 : 8,
      pixfmt == RGB24 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_GRAY,
      PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  if (pixfmt == GRAY16) png_set_swap(png);

  // Write image content.
  vector<png_bytep> rows(height);
  for (int i = 0; i < height; ++i) {
    rows[i] = (png_bytep) (((uint8_t*) data) + i * pitch);
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during writting png image " << filepath;
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_write_image(png, &rows[0]);
  png_write_end(png, NULL);

  fclose(fp);
  png_destroy_write_struct(&png, &info);
  return true;
}

}  // namespace

template <>
bool ReadImage<uint8_t>(const std::string& filepath, Image<uint8_t>* image) {
  return ReadImageGray8(filepath, image);
}

template <>
bool ReadImage<uint8_t, 1>(const std::string& filepath,
                           MCImage<uint8_t, 1>* image) {
  return ReadImageGray8(filepath, image);
}

template <>
bool ReadImage<uint8_t, 3>(const std::string& filepath,
                           MCImage<uint8_t, 3>* image) {
  return ReadImageRGB8(filepath, image);
}

bool ReadImageGray8(const string& filepath, MCImageGray8* mc_image) {
  ImageGray8 image;
  bool ret = ReadImageGray8(filepath, &image);
  if (ret) mc_image->SetPlane(0, image);
  return ret;
}

bool ReadImageGray16(const string& filepath, MCImageGray16* mc_image) {
  ImageGray16 image;
  bool ret = ReadImageGray16(filepath, &image);
  if (ret) mc_image->SetPlane(0, image);
  return ret;
}

bool WriteImageGray8(const MCImageGray8& mc_image, const string& filepath) {
  return WriteImageGray8(mc_image.GetPlane(), filepath);
}

bool ReadImageGray8(const string& filepath, ImageGray8* image) {
  string ext = GetFileExtension(filepath);
  bool ret = false;
  RawImageBuffer imgbuf;
  if (ext == "png") {
    ret = ReadRawImageBufferFromPNG(filepath, &imgbuf);
    VLOG(1) << "ReadImageGray8 " << imgbuf.pixfmt << "," << imgbuf.width << "x"
            << imgbuf.height << ", " << imgbuf.pitch;;
  }
  if (!ret) return false;
  switch (imgbuf.pixfmt) {
    case RGB24: {
      MCImageRGB8 rgb;
      rgb.FromBuffer(&imgbuf.data[0], imgbuf.width, imgbuf.height, 3);
      RGB8ToGray8(rgb, image);
      break;
    }
    case GRAY16: {
      LOG(WARNING) << "converting GRAY16 to GRAY8.";
      ImageGray16 gray16;
      gray16.FromBuffer(&imgbuf.data[0], imgbuf.width, imgbuf.height);
      image->FromArray((gray16 / 255).cast<uint8_t>(), gray16.width(),
                       gray16.height());
//      ImageGray8 tmp_image = (gray16 / 255):
//      *image = tmp_image;
      break;
    }
    case GRAY8:
      image->FromBuffer(&imgbuf.data[0], imgbuf.width, imgbuf.height);
      break;
    default:
      LOG(ERROR) << "unsupported pixel format " << imgbuf.pixfmt;
      return false;
  }
  VLOG(1) << "ReadImageGray8 " << image->rows() << "x" << image->cols();
  return ret;
}

bool ReadImageGray16(const std::string& filepath, ImageGray16* image) {
  string ext = GetFileExtension(filepath);
  bool ret = false;
  RawImageBuffer imgbuf;
  if (ext == "png") {
    ret = ReadRawImageBufferFromPNG(filepath, &imgbuf);
    VLOG(2) << "ReadImageGray16 " << imgbuf.pixfmt << "," << imgbuf.width << "x"
            << imgbuf.height << ", " << imgbuf.pitch;;
  }
  if (!ret) return false;
  switch (imgbuf.pixfmt) {
    case GRAY16:
      image->FromBuffer(&imgbuf.data[0], imgbuf.width, imgbuf.height);
      break;
    default:
      LOG(ERROR) << "unsupported pixel format " << imgbuf.pixfmt;
      return false;
  }
  VLOG(2) << "ReadImageGray16 " << image->rows() << "x" << image->cols();
  return ret;
}

bool WriteImageGray8(const ImageGray8& image, const string& filepath) {
  string ext = GetFileExtension(filepath);
  bool ret = false;
  if (ext == "png") {
    ret = WriteRawImageBufferToPNG(image.data(), GRAY8, image.width(),
                                   image.height(), image.width(), filepath);
    VLOG(1) << "WriteImageGray8 '" << filepath << "', "
            << image.rows() << "x" << image.cols();
  } else {
    LOG(ERROR) << "unsupported image type " << ext;
  }
  return ret;
}

bool ReadImageRGB8(const string& filepath, MCImageRGB8* image) {
  string ext = GetFileExtension(filepath);
  bool ret = false;
  RawImageBuffer imgbuf;
  if (ext == "png") {
    ret = ReadRawImageBufferFromPNG(filepath, &imgbuf);
    VLOG(2) << "ImageRGB8 " << imgbuf.pixfmt << "," << imgbuf.width << "x"
            << imgbuf.height << ", " << imgbuf.pitch;;
  }
  if (!ret) return false;
  switch (imgbuf.pixfmt) {
    case RGB24:
      image->FromBuffer(&imgbuf.data[0], imgbuf.width, imgbuf.height, 3);
      break;
    case GRAY8: {
      MCImageGray8 gray;
      gray.FromBuffer(&imgbuf.data[0], imgbuf.width, imgbuf.height, 1);
      Gray8ToRGB8(gray, image);
      break;
    }
    case GRAY16: {
      LOG(WARNING) << "converting GRAY16 to RGB8.";
      ImageGray16 gray16;
      gray16.FromBuffer(&imgbuf.data[0], imgbuf.width, imgbuf.height);
      image->SetAllPlanes((gray16 / 255).cast<uint8_t>());
      break;
    }
    default:
      LOG(ERROR) << "unsupported pixel format " << imgbuf.pixfmt;
      return false;
  }
  VLOG(2) << "ReadImageRGB8 " << image->rows() << "x" << image->cols();
  return ret;
}

bool WriteImageRGB8(const MCImageRGB8& image, const string& filepath) {
  string ext = GetFileExtension(filepath);
  bool ret = false;
  if (ext == "png") {
    ret = WriteRawImageBufferToPNG(image.data(), RGB24, image.width(),
                                   image.height(), image.width() * 3, filepath);
    VLOG(2) << "WriteImageRGB8 '" << filepath << "', "
            << image.rows() << "x" << image.cols();
  } else {
    LOG(ERROR) << "unsupported image type " << ext;
  }
  return ret;
}

}  // namespace rvslam

