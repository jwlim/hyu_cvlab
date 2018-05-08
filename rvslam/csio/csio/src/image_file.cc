// image_file.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include "image_file.h"

#include <stdint.h>
#include <stdio.h>

#include <png.h>
#include <glog/logging.h>

using namespace std;

namespace csio {

bool ReadImageFromPNG(const string& filepath,
                      int* pixf, int* width, int* height, string* data) {
  // Open the image file and check if it's a valid png file.
  FILE* fp = fopen(filepath.c_str(), "rb");
  if (!fp) {
    //LOG(ERROR) << "Failed to open file '" << filepath << "'";
    return false;
  }
  png_byte png_header[8];  // 8 is the maximum size that can be checked.
  fread(png_header, 1, 8, fp);
  if (png_sig_cmp(png_header, 0, 8)) {
    LOG(ERROR) << "Incorrect PNG header '" << filepath << "'";
    fclose(fp);
    return false;
  }
  // Initialize the png struct to load the content.
  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                           NULL, NULL, NULL);
  if (!png) {
    LOG(ERROR) << "png_create_read_struct failed '" << filepath << "'";
    fclose(fp);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    LOG(ERROR) << "png_create_info_struct failed '" << filepath << "'";
    png_destroy_read_struct(&png, NULL, NULL);
    fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during setting up png read '" << filepath << "'";
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

  *pixf = pix::GRAY8;
  if (ctype == PNG_COLOR_TYPE_RGB || ctype == PNG_COLOR_TYPE_RGB_ALPHA) {
    *pixf = pix::RGB8;
    if (bit_depth > 8)  png_set_strip_16(png);
  } else if (bit_depth > 8) {
    *pixf = pix::GRAY16;
    png_set_swap(png);  // LSB -> MSB
  }
  // png_set_interlace_handling(png);
  // png_read_update_info(png, info);

  int w = *width = png_get_image_width(png, info);
  int h = *height = png_get_image_height(png, info);
  int pitch = w * (*pixf == pix::RGB8 ? 3 : *pixf == pix::GRAY16 ? 2 : 1);
  VLOG(1) << w << "x" << h << ", " << pitch;
  data->resize(pitch * h);
  vector<png_bytep> rows(h);
  for (int i = 0; i < h; ++i) rows[i] = (png_bytep) &(data[i * pitch]);

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

bool WriteImageToPNG(const void* data, int pixf, int width, int height,
                     int pitch, const string& filepath) {
  if (width <= 0 || height <= 0) {
    LOG(ERROR) << "Empty image '" << filepath << "'";
    return false;
  }
//  ushort pixf = ib.pixf;
//  if (pix::is_bayer(pixf))
//    pixf = (pixf & pix::BAYER16_MASK)? pix::GRAY16 : pix::GRAY8;
//  if (pixf!=pix::GRAY8 && pixf!=pix::GRAY16 && pixf!=pix::RGB24)
//    HMSGRET(false, ("imwrite_png(%s): unsupported image format %d:'%s' (gray8,gray16,rgb24 only)", path, pixf, pix::pixf2str(pixf)));

  FILE *fp = fopen(filepath.c_str(), "wb");
  if (!fp) {
    //LOG(ERROR) << "Failed to open " << filepath << "'";
    return false;
  }
  // Initialize png structs.
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                            NULL, NULL, NULL);
  if (!png) {
    LOG(ERROR) << "png_create_write_struct failed '" << filepath << "'";
    fclose(fp);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (!info) {
    LOG(ERROR) << "png_create_info_struct failed '" << filepath << "'";
    png_destroy_write_struct(&png, NULL);
    fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during setting up png write '" << filepath << "'";
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_init_io(png, fp);

  // Write png header.
  if (setjmp(png_jmpbuf(png))) {
    LOG(ERROR) << "Error during writting png header '" << filepath << "'";
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return false;
  }
  png_set_compression_level(png, 3);
  png_set_compression_mem_level(png, 9);
  png_set_IHDR(
      png, info, width, height,
      pixf == pix::GRAY16 ? 16 : 8,
      pixf == pix::RGB8 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_GRAY,
      PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  if (pixf == pix::GRAY16) png_set_swap(png);

  // Write image content.
  vector<png_bytep> rows(height);
  for (int i = 0; i < height; ++i) rows[i] = ((png_bytep) data) + i * pitch;

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

}  // namespace csio

/*
#endif //USELIBPNG
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
#ifdef USEMAGICK

bool imread_magick(HImagebuf &ib, const char *path)
{
  try {
    Image mgkimg(path);
    Geometry size = mgkimg.size();
    const int w = size.width(), h = size.height();
    PixelPacket *px = mgkimg.getPixels(0,0,w,h);
//HMSG("type %d", mgkimg.type());

    ib.alloc(w, h, pix::RGB24);  // QuantumDepth : RGB48?
    HRGB *buf = (HRGB*) ib.buf;
    for (int i0=0, y=0; y<h; y++, i0+=ib.pitch)
      for (int x=0, i=i0; x<w; x++, i++)
        buf[i].set(px[i].red, px[i].green, px[i].blue);
  }
  catch (Exception &e) {
    HMSGRET(false, ("loadmagick(%s): %s", path, e.what()));
  }
  return true;
}

//----------------------------------------------------------------------------

bool imwrite_magick(const HImagebuf &ib, const char *path)
{
  const int w = ib.w, h = ib.h;
  if (w == 0 || h == 0)
    HMSGRET(false, ("imwrite_magick(%s): empty image (%dx%d)", path,w,h));
  const ushort pixf = ib.pixf;
  if (pixf!=pix::GRAY8 && pixf!=pix::GRAY16 && pixf!=pix::RGB24)
    HMSGRET(false, ("imwrite_png(%s): unsupported image format (gray8,gray16,rgb24 only)", path));

  try {
    Image magick(Geometry(w,h), Color(0,0,0));
    magick.modifyImage();
    PixelPacket *px = magick.getPixels(0,0,w,h);
    int x, y, i0, i, j;
    switch (pixsz) {
    case IM_GRAY8: {
      const double coef = 1/255.0;
      for (y=0, i0=0, j=0; y < h; y++, i0+=yi)
        for (x=0, i=i0; x < w; x++, i+=xi, j++)
          px[j] = ColorGray(coef*buf[i]);
    } break;

    case IM_GRAY16: {
      const double coef = (maxval <= 0)? 1/65535.0 : 1.0/maxval;
      for (y=0, i0=0, j=0; y < h; y++, i0+=yi)
        for (x=0, i=i0; x < w; x++, i+=xi, j++)
          px[j] = ColorGray(coef*((ushort*)buf)[i]);
    } break;

    case IM_RGB24: {
      HRGB *rgb = (HRGB*) buf;
      magick.type(TrueColorType);
      PixelPacket *px = magick.getPixels(0,0,w,h);
      const double coef = 1/255.0;
      for (y=0, i0=0, j=0; y < h; y++, i0+=yi)
        for (x=0, i=i0; x < w; x++, i+=xi, j++)
          px[j] = ColorRGB(coef*rgb[i].r, coef*rgb[i].g, coef*rgb[i].b);
    } break;
    }
    magick.syncPixels();
    magick.write(path);
  }
  catch (Exception &e) {
    HMSGRET(false, ("savemagick(%s): %s", path, e.what()));
  }
  return true;
}

#endif //USEMAGICK
//----------------------------------------------------------------------------
};
*/

