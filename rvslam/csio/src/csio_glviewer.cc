// csio_glviewer.cc
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
#include "csio_util.h"
#include "image_file.h"

#if defined(__APPLE__)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include <pthread.h>

using namespace std;

DEFINE_string(in, "-", "File path for csio::InputStream (- for stdin).");
DEFINE_string(cap, "./dump/capture_image_%04d.png", "File path for capturing the screen.");
DEFINE_bool(pause, false, "Pause the viewer at the first frame.");
DEFINE_int32(buffer_len, 300, "Number of frames kept in memory.");
DEFINE_int32(end, 100, "Number of frames kept in memory.");


namespace csio {
typedef vector<Frame> FrameArray;
}  // namespace csio

struct ViewerApp {
  int win_id, w, h;
  csio::InputStream* csin_ptr;
  map<uint32_t, csio::View*> views;
  vector<csio::FrameArray*> frame_arrays;
  int display_frame_idx;

  ViewerApp() : win_id(-1), w(0), h(0), csin_ptr(NULL), display_frame_idx(0) {}
  ~ViewerApp() { Cleanup(); };
  bool Setup(csio::InputStream& csin);
  void Cleanup();
  void InitializeGL();
  void UpdateLayout();
  void Redraw();
  void OnResize(int w, int h);
  bool CheckCSIO();

  void Capture(vector<char>* rgb_buf, bool image_only = false);
  void SetDisplayFrameIdx(int frame_idx) {
    if (frame_idx < 0) frame_idx = 0;
    if (frame_idx >= frame_arrays.size()) frame_idx = frame_arrays.size() - 1;
    display_frame_idx = frame_idx;
  }
};

ViewerApp the_viewer;  // Global viewer instance.

//-----------------------------------------------------------------------------

bool ViewerApp::Setup(csio::InputStream& csin) {
  Cleanup();
  const vector<csio::ChannelInfo>& channels = csin.channels();
  win_id = -1;
  csin_ptr = &csin;
  int u = 0, v = 0;
  for (int i = 0; i < channels.size(); ++i) {
    csio::View* view = csio::View::Setup(channels[i], u, v);
    if (view == NULL) {
      LOG(ERROR) << "failed to setup view " << i << "/" << channels.size();
    } else if (views.count(channels[i].id) > 0) {
      LOG(ERROR) << "chid " << channels[i].id << " (" << i << ") already setup.";
    } else {
      views[channels[i].id] = view;
      ++u;
    }
  }
  LOG(INFO) << views.size() << "/" << channels.size() << " views setup.";
  UpdateLayout();
  return !views.empty();
}

void ViewerApp::Cleanup() {
  if (win_id >= 0) glutDestroyWindow(win_id);
  for (map<uint32_t, csio::View*>::iterator it = views.begin();
       it != views.end(); ++it) {
    delete it->second;
  }
  views.clear();
  for (int i = 0; i < frame_arrays.size(); ++i) delete frame_arrays[i];
  frame_arrays.clear();
  csin_ptr = NULL;
  win_id = -1;
}

void ViewerApp::InitializeGL() {
  glutInitWindowSize(w, h);
  glutInitWindowPosition(0, 0);
  win_id = glutCreateWindow("CSIO GL viewer");
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(1.0);
  glDepthFunc(GL_LESS);
  glDepthMask(GL_FALSE);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_ALPHA_TEST);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_SCISSOR_TEST);
  glShadeModel(GL_FLAT);
  for (int i = 0; i < views.size(); ++i) views[i]->InitializeGL();
}

void ViewerApp::UpdateLayout() {
  int x = 0, y = 0;
  w = 0, h = 0;
  for (int i = views.size() - 1; i >= 0; --i) {
    csio::View* view = views[i];
    view->UpdateLayout(x, y);
    y += view->height();
    h += view->height();
    w = max(w, view->width());
  }
  LOG(INFO) << "UpdateLayout: " << views.size() << ", " << w << "x" << h;
  // TODO: Function <glutReshapeWindow> called with no current window defined.
  //glutReshapeWindow(w, h);
}

void ViewerApp::Redraw() {
//  glScissor(0, 0, w, h);
//  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  if (frame_arrays.empty() || frame_arrays.back() == NULL) return;
  const int frame_idx = frame_arrays.size() - 1 - display_frame_idx;
  csio::FrameArray* frame_array_ptr = frame_arrays[frame_idx];
  map<uint32_t, csio::View*>::iterator it;
  for (int i = 0; i < frame_array_ptr->size(); ++i) {
    csio::Frame& frame = frame_array_ptr->at(i);
    if ((it = views.find(frame.chid)) == views.end()) {
      LOG(INFO) << "unknown chid " << frame.chid << ".";
    } else {
      it->second->DrawFrame(&frame);
    }
  }
  for (it = views.begin(); it != views.end(); ++it) it->second->Redraw();
  glutSwapBuffers();
}

void ViewerApp::OnResize(int width, int height) {
//  glViewport(0, 0, width, height);
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, w, h, 0, -1.0f, 1.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

bool ViewerApp::CheckCSIO() {
  if (csin_ptr == NULL) {
    //LOG(ERROR) << "csin_ptr is NULL";
    return false;
  }

  csio::FrameArray* frame_array_ptr = new csio::FrameArray();

  if (csin_ptr->FetchSyncFrames(frame_array_ptr) == false) {
    //LOG(ERROR) << "failed to fetch a frame.";
    delete frame_array_ptr;
    csin_ptr = NULL;
    return false;
  }
//  LOG(INFO) << "fetched " << frame.chid << ", " << frame.buf.size();

  // Put the frame array into the buffer (keep the buffer length below
  // FLAGS_buffer_len).
  if (frame_arrays.size() > FLAGS_buffer_len - 1) {
    delete frame_arrays.front();
    frame_arrays.erase(frame_arrays.begin());
  }
  frame_arrays.push_back(frame_array_ptr);
  display_frame_idx = 0;
  Redraw();

 // if (FLAGS_cap.empty() == false) {  // Capture the window.
    char filename[1024];
    static int fileidx = 0;
    sprintf(filename, FLAGS_cap.c_str(), fileidx++);
    vector<char> rgb_buf;
    the_viewer.Capture(&rgb_buf);
    csio::WriteRGB8ToPNG(rgb_buf.data(), w, h, w * 3, filename);
  //}
  return true;
}

void ViewerApp::Capture(vector<char>* rgb_buf, bool image_only) {
  if (image_only) {
    rgb_buf->resize(w * (h/2) * 3);
  } else {
    rgb_buf->resize(w * h * 3);
  }

  if (w % 4 != 0) glPixelStorei(GL_PACK_ALIGNMENT, w % 2 ? 1 : 2);

  if (image_only) {
    glReadPixels(0, h/2, w, h/2, GL_RGB, GL_UNSIGNED_BYTE, rgb_buf->data());
  } else {
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, rgb_buf->data());
  }

  char* buf = rgb_buf->data();

  if (image_only) {
    for (int y = 0; y < h/4; ++y) {
     char* p0 = &buf[y * w * 3];
     char* p1 = &buf[(h/2 - 1 - y) * w * 3];
     for (int i = 0; i < w * 3; ++i, ++p0, ++p1) {
       char tmp = *p0;
       *p0 = *p1;
       *p1 = tmp;
     }
    } 
  } else {
    for (int y = 0; y < h/2; ++y) {
     char* p0 = &buf[y * w * 3];
     char* p1 = &buf[(h - 1 - y) * w * 3];
     for (int i = 0; i < w * 3; ++i, ++p0, ++p1) {
       char tmp = *p0;
       *p0 = *p1;
       *p1 = tmp;
     }
    } 
  }
}

//-----------------------------------------------------------------------------

void ResizeGLWindow(int w, int h) { the_viewer.OnResize(w, h); }

void Redraw() { the_viewer.Redraw(); }

void CheckCSIOForDrawing() { 
  if(the_viewer.CheckCSIO() == false) {
    LOG(INFO) << "Terminating csio gl viewer";
    //the_viewer.Cleanup();
  }
}

void HandleKeyInput(unsigned char key, int x, int y) {
  static int count = 0;
  static bool paused = FLAGS_pause;
  switch (key) {
    case 27:  // ESC key.
      the_viewer.Cleanup();
      exit(0);
    case ' ':
      paused = !paused;
      glutIdleFunc(paused ? NULL : &CheckCSIOForDrawing);
      break;
    case 13:  // Enter key.
      paused = true;
      glutIdleFunc(NULL);
      CheckCSIOForDrawing();
      break;
    case 'C':
    case 'c': {  // Capture the window
      const int w = the_viewer.w, h = the_viewer.h;
      vector<char> rgb_buf;
      the_viewer.Capture(&rgb_buf);
      char filename[1024];
      sprintf(filename, "./dump/capture_image_%04d.png", count++);
      csio::WriteRGB8ToPNG(rgb_buf.data(), w, h, w * 3, filename);
    } break;
    case 'i':
    case 'I': {
      // Capture only 2D image
      const int w = the_viewer.w, h = the_viewer.h;
      vector<char> rgb_buf;
      the_viewer.Capture(&rgb_buf, true);
      char filename[1024];
      sprintf(filename, "capture_image_%04d.png", count++);
      csio::WriteRGB8ToPNG(rgb_buf.data(), w, h, w * 3, filename);
      } break;
    case '[':
      the_viewer.SetDisplayFrameIdx(the_viewer.display_frame_idx + 1);
      the_viewer.Redraw();
      break;
    case ']':
      the_viewer.SetDisplayFrameIdx(the_viewer.display_frame_idx - 1);
      the_viewer.Redraw();
      break;
    case 'r':
    case 'R':
      glutReshapeWindow(the_viewer.w, the_viewer.h);
      break; 
    default: {
      const int h = the_viewer.h;
      bool handled = false;
      for (int i = 0; !handled && i < the_viewer.views.size(); ++i) {
        handled = the_viewer.views[i]->HandleKey(key, 0, x, h - y + 1);
      }
      if (!handled) {
        LOG(ERROR) << "unknown key: " << static_cast<int>(key)
            << "(" << key << ")";
      }
    }
  } 
}

void HandleSpecialKeyInput(int special, int x, int y) {
  const int h = the_viewer.h;
  bool handled = false;
  for (int i = 0; !handled && i < the_viewer.views.size(); ++i) {
    handled = the_viewer.views[i]->HandleKey(0, special, x, h - y + 1);
  }
  if (!handled) {
    LOG(ERROR) << "unknown special key: " << static_cast<int>(special);
  }
}

void HandleMouseInput(int button, int state, int x, int y) {
  const int h = the_viewer.h;
  for (int i = 0; i < the_viewer.views.size(); ++i) {
    if (the_viewer.views[i]->HandleMouse(button, state, x, h - y + 1)) return;
  }
  VLOG(1) << "Mouse: " << button << ", " << state << ", " << x << ", " << y;
}

void HandleMouseMotion(int x, int y) {
  const int h = the_viewer.h;
  for (int i = 0; i < the_viewer.views.size(); ++i) {
    if (the_viewer.views[i]->HandleMouse(-1, -1, x, h - y + 1)) return;
  }
  VLOG(1) << "Motion: " << x << ", " << y;
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "starting up...";

  // OS X requires GLUT to run on the main thread
  glutInit(&argc, argv);

  // Setup csio::InputStream.
  LOG(INFO) << "setting up csio::Inputstream (in=" << FLAGS_in << ").";
  csio::InputStream csin;
  if (csin.Setup(FLAGS_in) == false) {
    LOG(ERROR) << "failed to open csio::InputStream (in=" << FLAGS_in << ").";
    return -1;
  }
  if (the_viewer.Setup(csin) == false) {
    LOG(ERROR) << "failed to setup the viewer (in=" << FLAGS_in << ").";
    return -1;
  }
  LOG(INFO) << "setup csio::InputStream(" << FLAGS_in << ") complete.";

//  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
  glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH);
  the_viewer.InitializeGL();

  glutDisplayFunc(&Redraw);
  glutReshapeFunc(&ResizeGLWindow);
  glutKeyboardFunc(&HandleKeyInput);
  glutSpecialFunc(&HandleSpecialKeyInput);
  glutMouseFunc(&HandleMouseInput);
  glutMotionFunc(&HandleMouseMotion);

  if (FLAGS_pause) {
    CheckCSIOForDrawing();  // Call once.
  } else {
    glutIdleFunc(&CheckCSIOForDrawing);
  }
  LOG(INFO) << "enter GLUT main loop.";
  glutMainLoop();

  return 0;
}
