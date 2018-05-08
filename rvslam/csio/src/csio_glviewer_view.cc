// csio_glviewer_view.cc
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)

#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "csio_glviewer_view.h"
#include "csio_stream.h"
#include "csio_frame_parser.h"
#include "image_file.h"

#if defined(__APPLE__)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glu.h>
#endif
#include <pthread.h>

using namespace std;

DEFINE_double(fov, 90.0, "GL camera field of view");
DEFINE_string(cam, "", "GL camera parameters (x,y,z,rx,ry,rz).");
DEFINE_string(grid, "", "GL grid parameters (size,step,z).");

namespace {

inline void ColormapJet(int val, unsigned char* rgb) {  // val: 0 ~ 256*4-1
  unsigned char &r = rgb[0], &g = rgb[1], &b = rgb[2];
  if (val < 128) r = 0, g = 0, b = 128 + val;
  else if ((val -= 128) < 256)  r = 0, g = val, b = 255;
  else if ((val -= 256) < 256)  r = val, g = 255, b = 255 - val;
  else if ((val -= 256) < 256)  r = 255, g = 255 - val, b = 0;
  else  r = 255 - val, g = 0, b = 0;
}

struct GLGrid {
  int size, step;
  float z;

  GLGrid() : size(60), step(5), z(-1.f) {
    if (FLAGS_grid.empty() == false) {
      sscanf(FLAGS_grid.c_str(), "%d,%d,%f", &size, &step, &z);
    }
  }
  void DrawGrid() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBegin(GL_LINES);
    for (int i = -size; i <= size; i += step) {
      if (i == 0)  continue;
      else if (i % 5 == 0) glColor4f(0.4, 0.4, 0.4, 0.3);
      else  glColor4f(0.2, 0.2, 0.2, 0.1);
      glVertex3f(-size, z, i);
      glVertex3f( size, z, i);
      glVertex3f(         i, z, -size);
      glVertex3f(         i, z,  size);
    }
    glColor4f(0.5, 0.0, 0.0, 0.5);
    glVertex3f(-size, z, 0);
    glVertex3f( size, z, 0);
    glColor4f(0.0, 0.5, 0.0, 0.5);
    glVertex3f(0, z, -size);
    glVertex3f(0, z,  size);
    glEnd();
    glDisable(GL_BLEND);
  }
};

struct GLCam {
  double fov, near_clip, far_clip;
  double pose[6];
  double R[9];  // 3x3 matrix (column major).
  bool ortho;

  GLCam() : fov(FLAGS_fov), near_clip(0.1), far_clip(1000.0), ortho(false) {
    for (int i = 0; i < 6; ++i) pose[i] = 0.0;
    if (FLAGS_cam.empty()) {
      pose[0] = M_PI / 2;
      pose[4] = 10.0;
    } else {
      sscanf(FLAGS_cam.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf",
             &pose[3], &pose[4], &pose[5], &pose[0], &pose[1], &pose[2]);
    }
    UpdateR();
  }
  void SetTopView(const double z = 10.0) {
    for (int i = 0; i < 6; ++i) pose[i] = 0.0;
    pose[0] = M_PI / 2;
    pose[4] = z;
  }
  void SetFrontView(const double z = 2.0) {
    for (int i = 0; i < 6; ++i) pose[i] = 0.0;
    pose[4] = 0.1;
    pose[5] = -z;
  }
  void UpdateR() {  // Rodrigues' formula.
    const double* rot = pose;
    double th = ::sqrt(rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2]);
    if (th < 1e-6) {
      R[1] = R[2] = R[3] = R[5] = R[6] = R[7] = 0.0;
      R[0] = R[4] = R[8] = 1.0;
    } else {
      const double r1 = rot[0] / th, r2 = rot[1] / th, r3 = rot[2] / th;
      const double a = (th >= M_PI)? -1 : cos(th);
      const double b = (th >= M_PI)? 0 : sin(th);
      const double c = 1.0 - a;
      R[0] =     a + c*r1*r1, R[3] = -b*r3 + c*r2*r1, R[6] =  b*r2 + c*r3*r1;
      R[1] =  b*r3 + c*r2*r1, R[4] =     a + c*r2*r2, R[7] = -b*r1 + c*r3*r2;
      R[2] = -b*r2 + c*r3*r1, R[5] =  b*r1 + c*r3*r2, R[8] =     a + c*r3*r3;
    }
  }
  void LookAt(int w, int h, bool update_rot = false) {
    if (update_rot) UpdateR();
    if (ortho == false) {
      gluPerspective(fov, (float) w / h, near_clip, far_clip);
    } else {
//    const int w = _dp3D.w, h = _dp3D.h, f=_dp3D.fov;
//    const float z = _dp3D.eye[1]/10;
//  // this clipping is too close?
//  const float nearZeroClip = -300.0;
//  const float farZeroClip  = 1000.0;
//    glOrtho(-w/f*z, w/f*z, -h/f*z, h/f*z, nearZeroClip, farZeroClip);
    }
//  vec3d_t e = _dp3D.eye, l = e+_dp3D.R.col(2), u = _dp3D.R.col(1);
    const double *e = &pose[3], *u = &R[3], *l = &R[6];
    gluLookAt(e[0], e[1], e[2], e[0] + l[0], e[1] + l[1], e[2] + l[2],
              u[0], u[1], u[2]);
  }
};

}  // namespace

namespace csio {

//-----------------------------------------------------------------------------

class ViewImage : public View {
 public:
  ViewImage(const string& pixel_type, int w, int h, int maxv)
      : View(), pixel_type_(pixel_type), rgb_buf_(w * h * 3, 128),
        frame_ptr_(NULL), maxv_(maxv), jet_(true) {}
  virtual ~ViewImage() {}

  virtual void InitializeGL();
  virtual void DrawFrame(const csio::Frame* frame_ptr);
  virtual void Redraw();

 protected:
  void UpdateBuffer(const csio::Frame* frame_ptr);
//  void DrawBufferRGB();
  void DrawBufferDepthPointCloud();

  std::string pixel_type_;
  std::vector<char> rgb_buf_;
  const csio::Frame* frame_ptr_;
  int maxv_;
  bool jet_;
  GLuint tex_;
};

//-----------------------------------------------------------------------------

class View3D : public View {
 public:
  View3D(int w, int h)
      : View(), frame_ptr_(NULL), show_grid_(true), decoration_(true), cam_(),
      point_scaler(1.f) {}
  virtual ~View3D() {}

  virtual void InitializeGL();
  virtual void DrawFrame(const csio::Frame* frame_ptr);
  virtual void Redraw();
  virtual bool HandleKey(int key, int special, int x, int y);
  virtual bool HandleMouse(int button, int state, int x, int y);

 protected:
  std::vector<char> buf_;
  const csio::Frame* frame_ptr_;
  bool show_grid_;
  bool decoration_;
  GLCam cam_;
  GLGrid grid_;
  int last_x, last_y, last_button;
  float point_scaler;
};

//-----------------------------------------------------------------------------

View* View::Setup(const csio::ChannelInfo& ch_info, int u, int v) {
  map<string, string> cfg;
  string type = csio::ParseTypeStr(ch_info.type, &cfg);

  LOG(INFO) << "csio_type '" << ch_info.type << "' -> '" << type << "'";
  for (map<string, string>::const_iterator i = cfg.begin(); i != cfg.end(); ++i)
    LOG(INFO) << "cfg: '" << i->first << "' : '" << i->second << "'";

  string pixel_type;
  int w = 640, h = 480;
  View* view = NULL;
  if (csio::IsImageType(type, cfg, &pixel_type, &w, &h)) {
    LOG(INFO) << "csio_type '" << type << "' recognized:"
        << pixel_type << " (" << w << "x" << h << ").";
    int maxv = 0;
    if (cfg.count("maxv") > 0) maxv = atoi(cfg["maxv"].c_str());
    LOG(INFO) << "maxv: " << maxv;
    view = new ViewImage(pixel_type, w, h, maxv);
  } else if (csio::IsGeometricObjectType(type, cfg, &w, &h)) {
    LOG(INFO) << "csio_type '" << type << "' recognized: 3D("
        << w << "x" << h << ").";
    view = new View3D(w, h);
  } else {
    LOG(WARNING) << "unknown csio_type '" << type << "' - skipping.";
    return NULL;
  }
  view->info = ch_info;
  view->gl.u = u;
  view->gl.v = v;
  view->gl.w = w;
  view->gl.h = h;
  return view;
}

void View::UpdateLayout(int x, int y) {
  gl.x = x, gl.y = y;
}

// ----------------------------------------------------------------------------

void ViewImage::InitializeGL() {
  glGenTextures(1, &tex_);
  glBindTexture(GL_TEXTURE_2D, tex_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void ViewImage::UpdateBuffer(const csio::Frame* frame_ptr) {
  CHECK_NOTNULL(frame_ptr);
  if (pixel_type_ == "rgb8") {
    memcpy(rgb_buf_.data(), frame_ptr->buf.data(), frame_ptr->buf.size());
  } else if (pixel_type_ == "gray8") {
    // Do conversion
    const uint8_t* pixel_buf = reinterpret_cast<const
        uint8_t*>(frame_ptr->buf.data());
    uint8_t* rgb_buf = reinterpret_cast<uint8_t*>(rgb_buf_.data());
    for (int i = 0, j = 0; i < frame_ptr->buf.size(); ++i, j += 3) {
      rgb_buf[j] = rgb_buf[j+1] = rgb_buf[j+2] = pixel_buf[i];
    }
  } else if (pixel_type_ == "gray16") {
    const int n = frame_ptr->buf.size() / 2;
    const uint16_t* depth_buf =
        reinterpret_cast<const uint16_t*>(frame_ptr->buf.data());
    uint8_t* rgb_buf = reinterpret_cast<uint8_t*>(rgb_buf_.data());
    if (jet_) {
      const int jet_shift = (maxv_ <= 0) ? 2 : ceil(log(maxv_ / 1024) / log(2));
      for (int i = 0, j = 0; i < n; ++i, j += 3) {
        ColormapJet(depth_buf[i] >> jet_shift, &rgb_buf[j]);
      }
    } else {
      const int shift = (maxv_ <= 0) ? 2 : ceil(log(maxv_ / 256) / log(2));
      for (int i = 0, j = 0; i < n; ++i, j += 3) {
        rgb_buf[j] = rgb_buf[j + 1] = rgb_buf[j + 2] = (depth_buf[i] >> shift);
      }
    }
  } else {
    LOG(INFO) << "unknown pixel_type '" << pixel_type_ << "'";
  }
  frame_ptr_ = frame_ptr;
}

void ViewImage::DrawFrame(const csio::Frame* frame_ptr) {
  if (frame_ptr == NULL) return;
  if (frame_ptr != frame_ptr_) UpdateBuffer(frame_ptr);
  Redraw();
}

void ViewImage::Redraw() {
//  LOG(INFO) << "DrawBufferRGB: " << gl.x << "," << gl.y << ", "
//      << gl.w << "x" << gl.h << ", " << pixel_type_;
  glViewport(gl.x, gl.y, gl.w, gl.h);
  glScissor(gl.x, gl.y, gl.w, gl.h);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0.0, gl.w, 0.0, gl.h);
  glScalef(1, -1, 1);  // Change window coordinate (y+ = down)
  glTranslatef(0, -gl.h, 0);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glBindTexture(GL_TEXTURE_2D, tex_);
  if (gl.w % 4 != 0) glPixelStorei(GL_UNPACK_ALIGNMENT, gl.w % 2 ? 1 : 2);
  glTexImage2D(GL_TEXTURE_2D, 0, 3, gl.w, gl.h, 0, GL_RGB, GL_UNSIGNED_BYTE,
               rgb_buf_.data());
  glBegin(GL_TRIANGLE_FAN);
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glTexCoord2f(0, 0);  glVertex3f(0,    0,    0);
  glTexCoord2f(1, 0);  glVertex3f(gl.w, 0,    0);
  glTexCoord2f(1, 1);  glVertex3f(gl.w, gl.h, 0);
  glTexCoord2f(0, 1);  glVertex3f(0,    gl.h, 0);
  glEnd();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

/*
void ViewImage::DrawBufferDepthPointCloud() {
  LOG(INFO) << "DrawBufferDepthPointCloud: " << gl.x << "," << gl.y << ", "
      << gl.w << "x" << gl.h;
  glClearColor(1.f,1.f,1.f,0.f);
  glViewport(gl.x, gl.y, gl.w, gl.h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
//MSG("perspective %.3f %.3f", _dp3D.fov, (float)_dp3D.w/_dp3D.h);
  const float fov = 90;
  gluPerspective(fov, (float) gl.w / gl.h, 0.1, 1000.0);
//  vec3d_t e = _dp3D.eye, l = e+_dp3D.R.col(2), u = _dp3D.R.col(1);
  gluLookAt(0, 100, -100,  0, 0, 1,  0, 1, 0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  DrawGrid();
/ *
    HtMat<float,4,4> M;
    setsubmat(M, 0,0, rodrigues(HtVec<float,3>(_dp3D.rot)/180*M_PI));
    M(3,3) = 1;
    glLoadMatrixf(M.m);
    if (_dp3D.ch_depth >= 0) {
      glBegin(GL_POINTS);
      float *ptsbuf = (float*) _dp3D.ib_pts.rgb_buf_;
      HIMAGE_LOOPIF(_dp3D.img, HRGB, ptsbuf[3*i+2] > 0,
            glColor3ub(buf[i].r,buf[i].g,buf[i].b);  glVertex3fv(&ptsbuf[3*i]); );
      glEnd();
    }
* /
}
*/

// ----------------------------------------------------------------------------

void View3D::InitializeGL() {
  glClearColor(1.f, 1.f, 1.f, 0.f);
  glScissor(gl.x, gl.y, gl.w, gl.h);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}

void View3D::DrawFrame(const csio::Frame* frame_ptr) {
  if (frame_ptr == NULL) return;
  if (frame_ptr != frame_ptr_) {
    buf_.resize(frame_ptr->buf.size());
    memcpy(buf_.data(), frame_ptr->buf.data(), frame_ptr->buf.size());
    frame_ptr_ = frame_ptr;
  }
  Redraw();
}

void View3D::Redraw() {
  glClearColor(1.f, 1.f, 1.f, 0.f);
  glScissor(gl.x, gl.y, gl.w, gl.h);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  glViewport(gl.x, gl.y, gl.w, gl.h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  cam_.LookAt(gl.w, gl.h);
/*
  if (ortho_ == false) {
//MSG("perspective %.3f %.3f", fov_, (float) gl.w / gl.h);
    gluPerspective(fov_, (float) gl.w / gl.h, 0.1, 1000.0);
  } else {
//    const int w = _dp3D.w, h = _dp3D.h, f=_dp3D.fov;
//    const float z = _dp3D.eye[1]/10;
//  // this clipping is too close?
//  const float nearZeroClip = -300.0;
//  const float farZeroClip  = 1000.0;
//    glOrtho(-w/f*z, w/f*z, -h/f*z, h/f*z, nearZeroClip, farZeroClip);
  }
//  vec3d_t e = _dp3D.eye, l = e+_dp3D.R.col(2), u = _dp3D.R.col(1);
//  gluLookAt(e[0],e[1],e[2], l[0],l[1],l[2], u[0],u[1],u[2]);
  gluLookAt(0, 100, -100,  0, 0, 1,  0, 1, 0);
*/
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  if (show_grid_) grid_.DrawGrid();

  GeometricObject gobj;
  int bufidx = 0;
  while ((bufidx = ParseGeometricObject(buf_, bufidx, &gobj)) > 0) {
    GLenum type = GL_POINTS;
    switch (gobj.type) {
      case 'p':  type = GL_POINTS;  break;
      case 'l':  type = GL_LINES;  break;
      case 'L':  type = GL_LINE_STRIP;  break;
      case 'O':  type = GL_LINE_LOOP;  break;
      case 't':  type = GL_TRIANGLES;  break;
      case 'T':  type = GL_TRIANGLE_STRIP;  break;
      case 'F':  type = GL_TRIANGLE_FAN;  break;
      case 'q':  type = GL_QUADS;  break;
      case 'Q':  type = GL_QUAD_STRIP;  break;
      case 'P':  type = GL_POLYGON;  break;
      default:  LOG(FATAL) << "unknown type '" << static_cast<int>(gobj.type)
                << "' at " << bufidx;
    }
    GLfloat org_point_size = 0.f, org_line_width = 0.f;
    GLint line_stipple_factor = 1;
    if (decoration_) {
      for (int i = 0; i < gobj.opts.size(); ++i) {
        const GeometricObject::Opt& opt = gobj.opts[i];
        switch (opt.mode) {
          case GeometricObject::OPT_POINT_SIZE:
            glGetFloatv(GL_POINT_SIZE, &org_point_size);
            glPointSize(opt.value * point_scaler);
            break;
          case GeometricObject::OPT_LINE_WIDTH:
            glGetFloatv(GL_LINE_WIDTH, &org_line_width);
            glLineWidth(opt.value);
            break;
          case GeometricObject::OPT_LINE_STIPPLE_FACTOR:
            line_stipple_factor = opt.value;
            break;
          case GeometricObject::OPT_LINE_STIPPLE:
            glLineStipple(line_stipple_factor,
                          (static_cast<uint16_t>(opt.value) << 8) | opt.value);
            glEnable(GL_LINE_STIPPLE);
            break;
          default:
            LOG(ERROR) << "unknown opt '" << static_cast<int>(opt.mode) << "'";
        }
      }
    }
//    LOG(INFO) << "GeomObj: " << gobj.type << " (" << gobj.num_nodes
//        << "x" << gobj.node_size << ") " << bufidx << "/" << buf_.size();
//    if ((gobj.mode & 0x30)) {
//      const int f_mode = (gobj.mode & 0xf0);
//      glPolygonMode(GL_FRONT_AND_BACK, (gobj.mode == 0x20)? GL_POINT : GL_LINE);
//    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    for (int node_idx = 0; node_idx < gobj.num_nodes; ++node_idx) {
      glBegin(type);
      glColor4ubv(gobj.color(node_idx));  // Set node color.
      const float* pts = gobj.pts(node_idx);
      for (int pts_idx = 0; pts_idx < gobj.node_size; ++pts_idx) {
        const float* p = pts + 3 * pts_idx;
        glVertex3f(-p[0], -p[1], p[2]);
      }
      glEnd();
    }
    glDisable(GL_BLEND);

    // Reset options.
    if (decoration_) {
      for (int i = 0; i < gobj.opts.size(); ++i) {
        const GeometricObject::Opt& opt = gobj.opts[i];
        switch (opt.mode) {
          case GeometricObject::OPT_POINT_SIZE:
            glPointSize(org_point_size);
            break;
          case GeometricObject::OPT_LINE_WIDTH:
            glLineWidth(org_line_width);
            break;
          case GeometricObject::OPT_LINE_STIPPLE:
            glDisable(GL_LINE_STIPPLE);
            break;
        }
      }
    }

//    // Must follow glEnd().
//    if (stipple) glDisable(GL_LINE_STIPPLE);
//    if ((buf_mode & 0x30)) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

bool View3D::HandleKey(int key, int special, int x, int y) {
  if (!IsInside(x, y)) return false;
  static double step_tr = 5;
  const double step_rot = M_PI / 36;
  VLOG(1) << "HandleKey: " << key;
  switch (key) {
    case 'a':  cam_.pose[3] += step_tr;  break;
    case 'd':  cam_.pose[3] -= step_tr;  break;
    case 'q':  cam_.pose[4] += step_tr;  break;
    case 'e':  cam_.pose[4] -= step_tr;  break;
    case 'w':  cam_.pose[5] += step_tr;  break;
    case 's':  cam_.pose[5] -= step_tr;  break;
    case 'x':  cam_.pose[0] += step_rot;  break;
    case 'y':  cam_.pose[1] += step_rot;  break;
    case 'z':  cam_.pose[2] += step_rot;  break;
    case 'X':  cam_.pose[0] -= step_rot;  break;
    case 'Y':  cam_.pose[1] -= step_rot;  break;
    case 'Z':  cam_.pose[2] -= step_rot;  break;
    case 'g':  show_grid_ = !show_grid_;  break;
    case 'h':  decoration_ = !decoration_;  break;
    case 't':  cam_.SetTopView(cam_.pose[4]); break;
    case 'T':  cam_.SetTopView(100.0); break;
    case 'f':  cam_.SetFrontView(2.0); break;
    case 'F':  cam_.SetFrontView(10.0); break;
//    case '+':  point_scaler += 0.1; break;
//    case '-':  point_scaler -= 0.1; break;
    case '+':  step_tr += 1.0; break;
    case '-':  step_tr -= 1.0; break;
    default:  return false;
  }
  VLOG(1) << setfill(' ') << "Camera: "
      << cam_.pose[3] << "," << cam_.pose[4] << "," << cam_.pose[5] << ","
      << cam_.pose[0] << "," << cam_.pose[1] << "," << cam_.pose[2];
  LOG(INFO) << "step_tr = " << step_tr;
  cam_.UpdateR();
  Redraw();
  glFlush();
  return true;
}

bool View3D::HandleMouse(int button, int state, int x, int y) {
  if (IsInside(x, y)) {
    // state == 0 := clicked
    // state == 1 := released
    // state == -1 := dragging
    // button == 0 := left button
    // button == 2 := right button
    if(state == 0) {
      // Button pressed, log current coordinate.
      last_x = x; last_y = y; last_button = button;
    } else if(state == 1) {
      // Do something
    } else if(state == -1) {
      // Compute delta
      float dx = (x - last_x)/10.f; float dy = (y - last_y)/10.f;
      // While dragging, depending on last_button status, rotate or drag.
      if(last_button == 0) {
        // Drag
        cam_.pose[3] += dx;
        cam_.pose[5] -= dy;
      } else if (last_button == 2) {
        // Rotate
        cam_.pose[0] -= (dy);
        cam_.pose[1] -= (dx);
      }
      last_x = x; last_y = y;
    }
    VLOG(1) << setfill(' ') << "Camera: "
      << cam_.pose[3] << "," << cam_.pose[4] << "," << cam_.pose[5] << ","
      << cam_.pose[0] << "," << cam_.pose[1] << "," << cam_.pose[2];
    cam_.UpdateR();
    Redraw();
    glFlush();
 
    return true;
  }
  return false;
}

// ----------------------------------------------------------------------------

}  // namespace csio

