// csio_frame_parser.h
//
//  Channeled stream IO based on C++ standard library.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _CSIO_FRAME_PARSER_H_
#define _CSIO_FRAME_PARSER_H_

#include <string>
#include <vector>
#include "csio_channel.h"
#include "csio_util.h"

namespace csio {

inline std::string ParseTypeStr(const std::string& type_str,
                                std::map<std::string, std::string>* cfg) {
  std::string str = type_str;
  std::string type = StrTok(";", &str);
  if (!str.empty() && cfg) SplitStr(str, ";", "'\"", "=", cfg);
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

//------------------------------------------------------------------------------
// Image type - uncompressed image buffer.

inline bool IsImageType(const std::string& type,
                        const std::map<std::string, std::string>& cfg,
                        std::string* pixel_type = NULL,
                        int* width = NULL, int* height = NULL) {
  if (type.substr(0, 5) != "image" || cfg.count("pixel") == 0 ||
      (cfg.count("size") == 0 &&
       (cfg.count("width") == 0 || cfg.count("height") == 0))) return false;
  int w = -1, h = -1;
  if (cfg.count("size") > 0) {
    sscanf(cfg.find("size")->second.c_str(), "%dx%d", &w, &h);
  } else {
    w = atoi(cfg.find("width")->second.c_str());
    h = atoi(cfg.find("height")->second.c_str());
  }
  if (w < 0 || h < 0) return false;
  if (pixel_type) *pixel_type = cfg.find("pixel")->second;
  if (width) *width = w;
  if (height) *height = h;
  return true;
}

inline std::string MakeImageTypeStr(const std::string& pixel_type,
                                    int width, int height) {
  std::stringstream ss;
  ss << "image/x-csio-raw;pixel=" << pixel_type
      << ";size=" << width << "x" << height;
  return ss.str();
}

inline std::string MakeImageTypeStr(
    const std::string& pixel_type, int width, int height,
    const std::map<std::string, std::string>& cfg) {
  std::stringstream ss;
  ss << MakeImageTypeStr(pixel_type, width, height);
  for (std::map<std::string, std::string>::const_iterator it = cfg.begin();
       it != cfg.end(); ++it) {
    if (ContainsAnyOf(it->second, "; \t\r\n")) {
      ss << ";" << it->first << "=\"" << it->second << "\"";
    } else {
      ss << ";" << it->first << "=" << it->second;
    }
  }
  return ss.str();
}

//------------------------------------------------------------------------------
// Geometric 3D objects.
// Format: header(2+2k), node_size(4), num_nodes(4), { color(4), { pts(3*4) } }
//   header: type(1), num_opts(1), { mode(1), value(1) }
//   node_size: number of points in a node (uint32_t)
//   num_nodes: number of nodes (uint32_t)
//   color: r,g,b,a (bytes)
//   pts: x,y,z (floats)

struct GeometricObject {
  struct Opt { uint8_t mode, value; };
  char type;
  std::vector<Opt> opts;
  uint16_t reserved;
  uint32_t node_size, num_nodes;
  uint8_t* data;

  enum ObjectType { VIEW = 'v', POINTS = 'p',
    LINES = 'l', LINE_STRIP = 'L', LINE_LOOP = 'O',
    TRIANGLES = 't', TRIANGLE_STRIP = 'T', TRIANGLE_FAN = 'F',
    QUADS = 'q', QUAD_STRIP = 'Q', POLYGON = 'P' };
  enum OptMode { OPT_POINT_SIZE = 0x01, OPT_LINE_WIDTH = 0x02,
    OPT_LINE_STIPPLE = 0x03, OPT_LINE_STIPPLE_FACTOR = 0x04 };

  static Opt MakeOpt(uint8_t m, uint8_t v) {
    Opt opt;
    opt.mode = m, opt.value = v;
    return opt;
  }

  uint8_t* node_ptr(int node_idx = 0) const {
    return data + (3 * node_size + 1) * node_idx * 4;
  }
  uint8_t* color(int node_idx = 0) const { return node_ptr(node_idx); }
  void set_color(int node_idx,
                 uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) { 
    uint8_t* c = color(node_idx);
    c[0] = r, c[1] = g, c[2] = b, c[3] = a;
  }
  float* pts(int node_idx = 0) const {
    return reinterpret_cast<float*>(node_ptr(node_idx) + 4);
  }
  void set_point(int node_idx, int pt_idx, float x, float y, float z) {
    float* buf = pts(node_idx) + pt_idx * 3;
    buf[0] = x, buf[1] = y, buf[2] = z;
  }
};

inline bool IsGeometricObjectType(const std::string& type,
                                  const std::map<std::string, std::string>& cfg,
                                  int* w, int* h) {
  if (type != "model/x-csio-raw") return false;
  if (cfg.count("size") > 0) {
    sscanf(cfg.find("size")->second.c_str(), "%dx%d", w, h);
  } else {
    if (cfg.count("width") > 0) *w = atoi(cfg.find("width")->second.c_str());
    if (cfg.count("height") > 0) *h = atoi(cfg.find("height")->second.c_str());
  }
  return true;
}

inline std::string MakeGeometricObjectTypeStr(int w = 0, int h = 0) {
  std::stringstream ss;
  ss << "model/x-csio-raw";
  if (w > 0) ss << ";width=" << w;
  if (h > 0) ss << ";height=" << h;
  return ss.str();
}

inline int ParseGeometricObject(const std::vector<char>& frame_buf, int idx,
                                GeometricObject* obj) {
  if (idx < 0 || idx >= frame_buf.size()) return -1;
  const char* bufptr = &frame_buf[idx];
  const uint8_t* uint8_bufptr = reinterpret_cast<const uint8_t*>(bufptr);
  obj->type = bufptr[0];
  uint8_t num_opts = bufptr[1];
  obj->opts.resize(num_opts);
  for (int i = 0, j = 2; i < num_opts; ++i, j += 2) {
    obj->opts[i].mode = uint8_bufptr[j];
    obj->opts[i].value = uint8_bufptr[j + 1];
  }
  int offset = 2 + num_opts * 2;
  obj->node_size = *reinterpret_cast<const uint32_t*>(bufptr + offset);
  obj->num_nodes = *reinterpret_cast<const uint32_t*>(bufptr + offset + 4);
  obj->data = const_cast<uint8_t*>(uint8_bufptr + offset + 8);
  return idx + (obj->node_ptr(obj->num_nodes) - uint8_bufptr);
}

inline int ComputeGeometricObjectSize(uint8_t num_opts,
                                      uint32_t node_size, uint32_t num_nodes) {
  return 10 + 2 * num_opts + (3 * node_size + 1) * num_nodes * 4;
}

inline int ComputeGeometricObjectSize(uint32_t node_size, uint32_t num_nodes) {
  return ComputeGeometricObjectSize(0, node_size, num_nodes);
}

inline GeometricObject AddGeometricObjectToBuffer(
    char type, const std::vector<GeometricObject::Opt>& opts,
    uint32_t node_size, uint32_t num_nodes,
    std::vector<char>* buf) {
  // Make space in the buffer for the geometric object.
  int bufidx = buf->size();
  buf->resize(buf->size() +
              ComputeGeometricObjectSize(opts.size(), node_size, num_nodes));
  char* bufptr = &buf->at(bufidx);
  uint8_t* uint8_bufptr = reinterpret_cast<uint8_t*>(bufptr);
  GeometricObject obj;
  bufptr[0] = obj.type = type;
  uint8_bufptr[1] = (obj.opts = opts).size();
  for (int i = 0, j = 2; i < opts.size(); ++i, j += 2) {
    uint8_bufptr[j] = opts[i].mode;
    uint8_bufptr[j + 1] = opts[i].value;
  }
  int offset = 2 + opts.size() * 2;
  uint32_t* uint32_bufptr = reinterpret_cast<uint32_t*>(uint8_bufptr + offset);
  uint32_bufptr[0] = obj.node_size = node_size;
  uint32_bufptr[1] = obj.num_nodes = num_nodes;
  obj.data = uint8_bufptr + offset + 8;
  return obj;
}

inline GeometricObject AddGeometricObjectToBuffer(
    char type, uint32_t node_size, uint32_t num_nodes,
    std::vector<char>* buf) {
  std::vector<GeometricObject::Opt> opts;
  return AddGeometricObjectToBuffer(type, opts, node_size, num_nodes, buf);
}

//------------------------------------------------------------------------------
// IMU
inline bool IsIMUType(const std::string& type,
                      const std::map<std::string, std::string>& cfg) {
  if (type != "text/imu_pose") return false;
  return true;
}

inline std::string MakeIMUTypeStr() {
  std::stringstream ss;
  ss << "text/imu_pose";
  return ss.str();
}

}  // namespace csio
#endif  // _CSIO_FRAME_PARSER_H_
