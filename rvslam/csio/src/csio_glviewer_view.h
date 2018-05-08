// csio_glviewer_view.h
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _CSIO_GLVIEWER_VIEW_H_
#define _CSIO_GLVIEWER_VIEW_H_

#include <string>
#include <vector>

#include "csio_channel.h"
//#include "csio_stream.h"
//#include "csio_util.h"

#if defined(__APPLE__)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

namespace csio {

class View {
 public:
  virtual ~View() {}

  static View* Setup(const csio::ChannelInfo& ch_info, int u, int v);

  virtual void InitializeGL() = 0;
  virtual void DrawFrame(const csio::Frame* frame_ptr) = 0;
  virtual void Redraw() = 0;

  virtual bool HandleKey(int key, int special, int x, int y) { return false; }
  virtual bool HandleMouse(int button, int state, int x, int y) {
    return false;
  }

  void UpdateLayout(int x, int y);

  int width() const { return gl.w; }
  int height() const { return gl.h; }

 protected:
  View() {}

  bool IsInside(int x, int y) const {
    return gl.x <= x && x < gl.x + gl.w && gl.y <= y && y < gl.y + gl.h;
  }

  csio::ChannelInfo info;
  struct GLInfo {
    int u, v;  // Subwindow location.
    int x, y, w, h;
  } gl;
};

}  // namespace csio
#endif  // _CSIO_GLVIEWER_VIEW_H_

