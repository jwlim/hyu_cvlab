// simple_ply_viewer.cc
//
// Author : Hyon Lim <limhyon@gmail.com>

#if defined(__APPLE__)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <iostream>
#include <vector>
#include <cstdio>
#include <string.h>
#include <math.h>
#include "zpr.h"
#include "image_file.h"

#include <gflags/gflags.h>

using namespace std;
using namespace csio;

const float DEG2RAD = 3.141593f / 180;

#define ENABLE_CONTROL_RECORDING // record key/mouse input for offline rendering into a movie

DEFINE_int32(fps, 25, "Frame per second");
DEFINE_double(radius, 3.0, "Turn radius");
DEFINE_double(org_depth, 4.0, "Depth of origin");
DEFINE_double(y_derotate, -40.0, "y derotate");

DEFINE_double(fov, 90.0, "GL camera field of view");
DEFINE_string(cam, "", "GL camera parameters (x,y,z,rx,ry,rz).");

// Drawing primitives
struct Point3f {
    float x, y, z;
};

struct Color3ub {
    unsigned char r, g, b;
};

struct Face {
    unsigned int a, b, c;
};

struct GLCam {
  double fov, near_clip, far_clip;
  double pose[6];
  double R[9];  // 3x3 matrix (column major).
  bool ortho;

  GLCam() : fov(FLAGS_fov), near_clip(-10.0), far_clip(10.0), ortho(true) {
    for (int i = 0; i < 6; ++i) pose[i] = 0.0;
    if (FLAGS_cam.empty()) {
    //pose[2] = M_PI;
      pose[5] = 3.0;
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
      double left = -(double)w/(double)h;
      double right = -left;
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glOrtho(left, right, -1.0, 1.0, near_clip, far_clip);
      glMatrixMode(GL_MODELVIEW);
    }
    const double *e = &pose[3], *u = &R[3], *l = &R[6];
    gluLookAt(e[0], e[1], e[2], e[0] + l[0], e[1] + l[1], e[2] + l[2],
              u[0], u[1], u[2]);
  }

  void LookAt(int w, int h, double depth, bool update_rot = false) {
    if (update_rot) UpdateR();
    const double *e = &pose[3], *u = &R[3], *l = &R[6];
    std::cout << e[0] << "," << e[1] << "," << e[2] << ",";
    std::cout << e[0] + depth*l[0] << "," <<e[1] + depth*l[1] << "," << e[2] +
        depth*l[2] << ",";
    std::cout << u[0] << "," << u[1] << "," << u[2] << std::endl;
    //gluLookAt(e[0], e[1], e[2], e[0] + depth*l[0], e[1] + depth*l[1], e[2] + depth*l[2], u[0], u[1], u[2]);
    double eye[3];
    eye[0] = -(e[0] + depth*l[0]);
    eye[1] = -(e[1] + depth*l[1]);
    eye[2] = -(e[2] + depth*l[2]);
    std::cout << eye[0] << "," << eye[1] << "," << eye[2] << std::endl;
    gluLookAt(eye[0], eye[1], eye[2], 0, 0, 0, u[0], u[1], u[2]);
  }
};

void drawFrustum(float fovY, float aspectRatio, float nearPlane, float farPlane);
void draw_cameras(void);
void draw_axes(void);
void drawAxis(float size);
void display(void);
void idle(void);
void keyboard(unsigned char key, int x, int y);
void load_ply(const char *filename, vector <Point3f> &points, vector <Color3ub> &colors, vector <Face> &faces);
void load_playback(const char *filename, double *projection_matrix, vector <double*> &states);

vector <Point3f> g_points;
vector <Color3ub> g_colors;
vector <Face> g_faces;
vector <double*> g_gl_states;

int g_time = 0;
int g_timebase = 0;
bool g_playback = false;
bool g_record_start = false;
bool g_capture = false;
bool g_motion = false;
bool g_camfollow = false;
bool g_save = false;
double g_projection_matrix[16];

#define NUM_CAMERA_PARAMS 9
#define POLY_INVERSE_DEGREE 6

typedef struct {
    double R[9];     /* Rotation */
    double t[3];     /* Translation */
    double f;        /* Focal length */
    double k[2];     /* Undistortion parameters */
    double k_inv[POLY_INVERSE_DEGREE]; /* Inverse undistortion parameters */
    char constrained[NUM_CAMERA_PARAMS];
    double constraints[NUM_CAMERA_PARAMS];  /* Constraints (if used) */
    double weights[NUM_CAMERA_PARAMS];      /* Weights on the constraints */
    double K_known[9];  /* Intrinsics (if known) */
    double k_known[5];  /* Distortion params (if known) */

    char fisheye;            /* Is this a fisheye image? */
    char known_intrinsics;   /* Are the intrinsics known? */
    double f_cx, f_cy;       /* Fisheye center */
    double f_rad, f_angle;   
    
#define NUM_CAMERA_PARAMS 9
#define POLY_INVERSE_DEGREE 6/* Other fisheye parameters */
    double f_focal;          /* Fisheye focal length */

    double f_scale, k_scale; /* Scale on focal length, distortion params */
} camera_params_t;

typedef struct
{
    double pos[3];
    double color[3];
} point_t;

// Bundle file.
std::vector<camera_params_t> cameras;
std::vector<point_t> points;
double bundle_version;

// Camera
GLCam g_cam;

// Color
int background_color = 0; // black, 1=white

void ReadBundleFile(char *bundle_file,
                    std::vector<camera_params_t> &cameras,
                    std::vector<point_t> &points, double &bundle_version) {
  FILE *f = fopen(bundle_file, "r");
  if (f == NULL) {
    printf("Error opening file %s for reading\n", bundle_file);
    return;
  }

  char *dummy = NULL;
  int num_reads;
  int num_images, num_points;

  char first_line[256];
  dummy = fgets(first_line, 256, f);
  if (first_line[0] == '#') {
    double version;
    sscanf(first_line, "# Bundle file v%lf", &version);

    bundle_version = version;
    printf("[ReadBundleFile] Bundle version: %0.3f\n", version);

    num_reads = fscanf(f, "%d %d\n", &num_images, &num_points);
  } else if (first_line[0] == 'v') {
      double version;
      sscanf(first_line, "v%lf", &version);
      bundle_version = version;
      printf("[ReadBundleFile] Bundle version: %0.3f\n", version);

      num_reads = fscanf(f, "%d %d\n", &num_images, &num_points);
  } else {
      bundle_version = 0.1;
      sscanf(first_line, "%d %d\n", &num_images, &num_points);
  }

  printf("[ReadBundleFile] Reading %d images and %d points...\n", num_images, num_points);

  /* Read cameras */
  for (int i = 0; i < num_images; i++) {
    double focal_length, k0, k1;
    double R[9];
    double t[3];

    if (bundle_version < 0.2) {
      /* Focal length */
      num_reads = fscanf(f, "%lf\n", &focal_length);
    } else {
      num_reads = fscanf(f, "%lf %lf %lf\n", &focal_length, &k0, &k1);
    }

    /* Rotation */
    num_reads = fscanf(f, "%lf %lf %lf\n%lf %lf %lf\n%lf %lf %lf\n",
           R+0, R+1, R+2, R+3, R+4, R+5, R+6, R+7, R+8);
    /* Translation */
    num_reads = fscanf(f, "%lf %lf %lf\n", t+0, t+1, t+2);

    // if (focal_length == 0.0)
    //     continue;

    camera_params_t cam;

    cam.f = focal_length;
    memcpy(cam.R, R, sizeof(double) * 9);
    memcpy(cam.t, t, sizeof(double) * 3);

    cameras.push_back(cam);
  }

  /* Read points */
  for (int i = 0; i < num_points; i++) {
    point_t pt;
    /* Position */
    num_reads = fscanf(f, "%lf %lf %lf\n", pt.pos + 0, pt.pos + 1, pt.pos + 2);
    /* Color */
    num_reads = fscanf(f, "%lf %lf %lf\n", pt.color + 0, pt.color + 1, pt.color + 2);

    int num_visible;
    num_reads = fscanf(f, "%d", &num_visible);

    for (int j = 0; j < num_visible; j++) {
      int view, key;
      num_reads = fscanf(f, "%d %d", &view, &key);

      double x, y;
      if (bundle_version >= 0.3)
        num_reads = fscanf(f, "%lf %lf", &x, &y);
    }

    if (num_visible > 0) {
      points.push_back(pt);
    }
  }

  fclose(f);
}

void drawAxis(float size)
{
    glDepthFunc(GL_ALWAYS);     // to avoid visual artifacts with grid lines
    glPushMatrix();             //NOTE: There is a bug on Mac misbehaviours of
                                //      the light position when you draw GL_LINES
                                //      and GL_POINTS. remember the matrix.

    // draw axis
    glLineWidth(3);
    glBegin(GL_LINES);
        glColor3f(1, 0, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(size, 0, 0);
        glColor3f(0, 1, 0);
        glVertex3f(0, 0, 0);
        glVertex3f(0, size, 0);
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, 0);
        glVertex3f(0, 0, size);
    glEnd();
    glLineWidth(1);

    // draw arrows(actually big square dots)
    glPointSize(5);
    glBegin(GL_POINTS);
        glColor3f(1, 0, 0);
        glVertex3f(size, 0, 0);
        glColor3f(0, 1, 0);
        glVertex3f(0, size, 0);
        glColor3f(0, 0, 1);
        glVertex3f(0, 0, size);
    glEnd();
    glPointSize(1);

    // restore default settings
    glPopMatrix();
    glDepthFunc(GL_LEQUAL);
}

void drawGrid(float size, float step)
{
    glBegin(GL_LINES);

    glColor3f(0.3f, 0.3f, 0.3f);
    for(float i=step; i <= size; i+= step)
    {
        glVertex3f(-size, 0,  i);   // lines parallel to X-axis
        glVertex3f( size, 0,  i);
        glVertex3f(-size, 0, -i);   // lines parallel to X-axis
        glVertex3f( size, 0, -i);

        glVertex3f( i, 0, -size);   // lines parallel to Z-axis
        glVertex3f( i, 0,  size);
        glVertex3f(-i, 0, -size);   // lines parallel to Z-axis
        glVertex3f(-i, 0,  size);
    }

    // x-axis
    glColor3f(0.5f, 0, 0);
    glVertex3f(-size, 0, 0);
    glVertex3f( size, 0, 0);

    // z-axis
    glColor3f(0,0,0.5f);
    glVertex3f(0, 0, -size);
    glVertex3f(0, 0,  size);

    glEnd();
}

void initLights()
{
    // set up light colors (ambient, diffuse, specular)
    GLfloat lightKa[] = {0.5f, 0.5f, 0.5f, 0.5f};      // ambient light
    GLfloat lightKd[] = {.9f, .9f, .9f, 1.0f};      // diffuse light
    GLfloat lightKs[] = {1, 1, 1, 1};               // specular light
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

    // position the light in eye space
    float lightPos[4] = {-1, 1, 1, 0};               // directional light
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    glEnable(GL_LIGHT0);                            // MUST enable each light source after configuration
}

void init() {
    glShadeModel(GL_SMOOTH);                        // shading mathod: GL_SMOOTH or GL_FLAT
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);          // 4-byte pixel alignment

    // enable/disable features
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    //glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    //glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_SCISSOR_TEST);

     // track material ambient and diffuse from surface color, call it before glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);

    glClearStencil(0);                              // clear stencil buffer
    glClearDepth(1.0f);                             // 0 is near, 1 is far
    glDepthFunc(GL_LEQUAL);

    //initLights();
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  if(argc < 3) {
    printf("./simple_ply_viewer model.ply bundle.out [playback.bin]\n");
    return 0;
  }

  /* Read the bundle file */
  if(argc >= 3) {
    ReadBundleFile(argv[2], cameras, points, bundle_version);
  }

  if(argc > 3) {
    load_playback(argv[3], g_projection_matrix, g_gl_states);
  }

  /* Initialise GLUT and create a window */
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(1024, 768);
  glutCreateWindow("GLT Mouse Zoom-Pan-Rotate");

  init();

  /* Configure GLUT callback functions */
  load_ply(argv[1], g_points, g_colors, g_faces);
  //glScalef(0.25,0.25,0.25);
  //g_cam.LookAt(640,480);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0, 0, 0,   // Eye
            0, 0, 1,   // Origin
            0, -1, 0);  // Upvector, -z.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  // Zpr. reshape will be called, and matrix mode set to MODELVIEW.
  zprInit();
  zprSelectionFunc(draw_axes);     /* Selection mode draw function */

  // Callback function add.
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutIdleFunc(idle);

  glutMainLoop();

  return 0;
}

void idle(void) {
  if(g_playback) {
    int width = glutGet(GLUT_WINDOW_WIDTH );
    int height = glutGet(GLUT_WINDOW_HEIGHT);

    vector<char> img;
    img.resize(width*height*3);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd(g_projection_matrix);

    glMatrixMode(GL_MODELVIEW);

    for(unsigned int i=0; i < g_gl_states.size(); i++) {
      glLoadMatrixd(g_gl_states[i]);
      display();
      if(width % 4 != 0) glPixelStorei(GL_PACK_ALIGNMENT, width % 2 ? 1 : 2);
      glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data());
      char file[512];
      sprintf(file, "frame-%06d.png", i);

      char* buf = img.data();
        for (int y = 0; y < height / 2; ++y) {
          char* p0 = &buf[y * width * 3];
          char* p1 = &buf[(height - 1 - y) * width * 3];
          for (int i = 0; i < width * 3; ++i, ++p0, ++p1) {
            char tmp = *p0;
            *p0 = *p1;
            *p1 = tmp;
          }
        }
        WriteRGB8ToPNG(img.data(), width, height, width * 3, file);
    }
    exit(0);
  } else if (g_capture) {
    int width = glutGet(GLUT_WINDOW_WIDTH );
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    vector<char> img;
    img.resize(width*height*3);
    if(width % 4 != 0) glPixelStorei(GL_PACK_ALIGNMENT, width % 2 ? 1 : 2);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data());
    char* buf = img.data();
      for (int y = 0; y < height / 2; ++y) {
        char* p0 = &buf[y * width * 3];
        char* p1 = &buf[(height - 1 - y) * width * 3];
        for (int i = 0; i < width * 3; ++i, ++p0, ++p1) {
          char tmp = *p0;
          *p0 = *p1;
          *p1 = tmp;
        }
      }
    WriteRGB8ToPNG(img.data(), width, height, width * 3, "capture.png");
    std::cout << "Captured. see capture.png" << std::endl;
    g_capture = false;
  }

  if (g_camfollow) {
    int width = glutGet(GLUT_WINDOW_WIDTH );
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    vector<char> img;
    img.resize(width*height*3);
    int index = 0;
    for (int i = 1; i < cameras.size(); i++) {
      const double *tc = cameras[i].t, *rc = cameras[i].R, *tp = cameras[i-1].t,
            *rp = cameras[i-1].R;
      const int N = 5;
      double t[3];
      double u[3];
      double v = 0;
      // Interpolated view.
      for (int k = 0; k < N; k++) {
        std::cout << i << "/" << cameras.size() << ", k= " << k << "/" << N << "\n";
        v = (double)k / (double)N; 
        //v = 0.5 - cos(-v * M_PI) * 0.5;
        t[0] = ((tc[0] * v) + (tp[0] * (1 - v)));
        t[1] = ((tc[1] * v) + (tp[1] * (1 - v)));
        t[2] = ((tc[2] * v) + (tp[2] * (1 - v)));
        u[0] = ((rc[1] * v) + (rp[1] * (1 - v)));
        u[1] = ((rc[4] * v) + (rp[4] * (1 - v)));
        u[2] = ((rc[7] * v) + (rp[7] * (1 - v)));

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(t[0], t[1], t[2], 0, 0, 0, -u[0], -u[1], -u[2]);
        display();

        if(g_save) {
          if(width % 4 != 0) glPixelStorei(GL_PACK_ALIGNMENT, width % 2 ? 1 : 2);
          glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data());
          char file[512];
          sprintf(file, "keyview-%06d.png", index++);

          char* buf = img.data();
          for (int y = 0; y < height / 2; ++y) {
            char* p0 = &buf[y * width * 3];
            char* p1 = &buf[(height - 1 - y) * width * 3];
              for (int i = 0; i < width * 3; ++i, ++p0, ++p1) {
                char tmp = *p0;
                *p0 = *p1;
                *p1 = tmp;
              }
          }
          WriteRGB8ToPNG(img.data(), width, height, width * 3, file);
        }
      }
    }
    g_camfollow = false;
  }

  if (g_motion) {
    int width = glutGet(GLUT_WINDOW_WIDTH );
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    vector<char> img;
    img.resize(width*height*3);
    int index = 0;

    for (int i = 0; i < 361; ++i) {
      std::cout << i << "/360\n";
      glMatrixMode(GL_MODELVIEW);
      glRotatef(1,0,1,0);
      display();
      if(g_save) {
        if(width % 4 != 0) glPixelStorei(GL_PACK_ALIGNMENT, width % 2 ? 1 : 2);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data());
        char file[512];
        sprintf(file, "keyview-%06d.png", index++);

        char* buf = img.data();
        for (int y = 0; y < height / 2; ++y) {
          char* p0 = &buf[y * width * 3];
          char* p1 = &buf[(height - 1 - y) * width * 3];
            for (int i = 0; i < width * 3; ++i, ++p0, ++p1) {
              char tmp = *p0;
              *p0 = *p1;
              *p1 = tmp;
            }
        }
        WriteRGB8ToPNG(img.data(), width, height, width * 3, file);
      } else {
        std::cout << "Not saving. to save, press s\n";
      }
    } 
    g_motion = false;
  }

#if 0
  if (g_motion) {
    int width = glutGet(GLUT_WINDOW_WIDTH );
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    vector<char> img;
    img.resize(width*height*3);
    int index = 0;
    const double r = FLAGS_radius;
    const double th = FLAGS_y_derotate * M_PI / 180.0;
    // Interpolated view.
    for (double k = 0; k < 2*M_PI; k+= M_PI/180) {
      double z = r * (1.0 - cos(k));
      double x = r * sin(k);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      double a = cos(th) * x;
      double b = -sin(th) * x;
      std::cout << a << ", " << b << "\n";
      gluLookAt(x, z*sin(th), z*cos(th), 0, 0, FLAGS_org_depth, 0, -cos(th), sin(th));
      display();

      if(g_save) {
        if(width % 4 != 0) glPixelStorei(GL_PACK_ALIGNMENT, width % 2 ? 1 : 2);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, img.data());
        char file[512];
        sprintf(file, "keyview-%06d.png", index++);

        char* buf = img.data();
        for (int y = 0; y < height / 2; ++y) {
          char* p0 = &buf[y * width * 3];
          char* p1 = &buf[(height - 1 - y) * width * 3];
            for (int i = 0; i < width * 3; ++i, ++p0, ++p1) {
              char tmp = *p0;
              *p0 = *p1;
              *p1 = tmp;
            }
        }
        WriteRGB8ToPNG(img.data(), width, height, width * 3, file);
      } else {
        std::cout << "Not saving. to save, press s\n";
      }
    }
    g_motion = false;
  }
#endif
  glutPostRedisplay();
}

void load_playback(const char *filename,
                   double *projection_matrix,
                   vector <double*> &states) {
  FILE *fp = fopen(filename, "r");

  if(!fp) {
    fprintf(stderr, "Error opening %s\n", filename);
    exit(1);
  }

  size_t reads;
  reads = fread((void*)projection_matrix, sizeof(double)*16, 1, fp);

  while(!feof(fp)) {
    double *m = new double[16];
    reads = fread((void*)&m[0], sizeof(double)*16, 1, fp);
    g_gl_states.push_back(m);
  }

  fclose(fp);
  printf("Playback images %d\n", (int)g_gl_states.size());
  g_playback = true;
}

void display(void) {
  int width = glutGet(GLUT_WINDOW_WIDTH );
  int height = glutGet(GLUT_WINDOW_HEIGHT);

  if (background_color == 0) {
    glClearColor(0.f, 0.f, 0.f, 0.f);
  } else if (background_color == 1) {
    glClearColor(1.f, 1.f, 1.f, 0.f);
  }

  // Automatically assumed MODELVIEW mode.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  drawGrid(10, 1);

  glPushMatrix();
  g_cam.LookAt(width, height, true);
  draw_cameras();
  //draw_axes();
  drawAxis(0.2);

  glBegin(GL_TRIANGLES);
  for(unsigned int i=0; i < g_faces.size(); i++) {
    int a = g_faces[i].a;
    int b = g_faces[i].b;
    int c = g_faces[i].c;

    glColor3ub(g_colors[a].r, g_colors[a].g, g_colors[a].b);
    glVertex3f(g_points[a].x, g_points[a].y, g_points[a].z);

    glColor3ub(g_colors[b].r, g_colors[b].g, g_colors[b].b);
    glVertex3f(g_points[b].x, g_points[b].y, g_points[b].z);

    glColor3ub(g_colors[c].r, g_colors[c].g, g_colors[c].b);
    glVertex3f(g_points[c].x, g_points[c].y, g_points[c].z);
  }
  glEnd();

  glBegin(GL_POINTS);
  for(unsigned int i=0; i < g_points.size(); ++i) {
    glColor3ub(g_colors[i].r, g_colors[i].g, g_colors[i].b);
    glVertex3f(g_points[i].x, g_points[i].y, g_points[i].z);
  }
  glEnd();

  glLineWidth(2.5);
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_LINES);
  for (unsigned int i = 1; i < cameras.size(); ++i) {
    glVertex3f(cameras[i-1].t[0], cameras[i-1].t[1], cameras[i-1].t[2]);
    glVertex3f(cameras[i].t[0], cameras[i].t[1], cameras[i].t[2]);
  }
  glEnd();

#ifdef ENABLE_CONTROL_RECORDING
  if(!g_playback && g_record_start) {
    g_time = glutGet(GLUT_ELAPSED_TIME);

    if ( (g_time - g_timebase) > (1000 / FLAGS_fps) ) {
      double *m = new double[16];
      glGetDoublev(GL_MODELVIEW_MATRIX, m);
#if 1
      printf("\n");
      printf("%f %f %f %f\n", m[0], m[1], m[2], m[3]);
      printf("%f %f %f %f\n", m[4], m[5], m[6], m[7]);
      printf("%f %f %f %f\n", m[8], m[9], m[10], m[11]);
      printf("%f %f %f %f\n", m[12], m[13], m[14], m[15]);
#endif
      g_gl_states.push_back(m);
      g_timebase = g_time;
    }
  }
#endif
  glPopMatrix();

  glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
  // Press ESC key.
  if (key == 27) {
    FILE *fp = fopen("playback.bin", "wb+");
    if(!fp) {
      fprintf(stderr, "Error writing playback file\n");
      exit(1);
    }
    // Write the projection matrix
    double m[16];
    glGetDoublev(GL_PROJECTION_MATRIX, m);
    fwrite((void*)&m[0], sizeof(double)*16, 1, fp);

    // Write the modelview matrix
    for(unsigned int i=0; i < g_gl_states.size(); i++) {
      fwrite((void*)&g_gl_states[i][0], sizeof(double)*16, 1, fp);
    }

    fclose(fp);

		exit(0);
  } else if (key == 'c') {
    // Self capture
    g_capture = true;
  } else if (key == 'r') {
    if (g_record_start == false) {
      std::cout << "Record started\n";
      g_record_start = true;
    } else {
      std::cout << "Record stopped\n";
      g_record_start = false;
    }
  } else if (key == 'm') {
    g_motion = true;
  } else if (key == 'o') {
    g_camfollow = true;
  } else if (key == 'v') {
    if (g_save == false)
      g_save = true;
    else
      g_save = false;
  }

  if (key == '1') {
    std::cout << "Black " << key <<  "\n";
    background_color = 0;
  } 
  
  if (key == '2') {
    std::cout << "White\n";
    background_color = 1;
  }

  const double step_rot = M_PI / 360;
  const double step_tr = 0.2;
  switch(key) {
    case 'x': g_cam.pose[0] += step_rot; break;
    case 'X': g_cam.pose[0] -= step_rot; break;
    case 'y': g_cam.pose[1] += step_rot; break;
    case 'Y': g_cam.pose[1] -= step_rot; break;
    case 'z': g_cam.pose[2] += step_rot; break;
    case 'Z': g_cam.pose[2] -= step_rot; break;
    case 'd': g_cam.pose[3] -= step_tr; break;
    case 'a': g_cam.pose[3] += step_tr; break;
    case 'e': g_cam.pose[4] -= step_tr; break;
    case 'q': g_cam.pose[4] += step_tr; break;
    case 's': g_cam.pose[5] -= step_tr; break;
    case 'w': g_cam.pose[5] += step_tr; break;
  }

  glutPostRedisplay();
}

void draw_cameras(void) {
  GLfloat m[16];
  for (unsigned int i = 0; i < cameras.size(); ++i) {
    glPushMatrix();
    const double *t = cameras[i].t, *r = cameras[i].R;
#define M(row,col)  m[col*4+row]
    M(0, 0) = r[0]; M(0, 1) = r[1]; M(0, 2) = r[2]; M(0, 3) = 0.0;
    M(1, 0) = r[3]; M(1, 1) = r[4]; M(1, 2) = r[5]; M(1, 3) = 0.0;
    M(2, 0) = r[6]; M(2, 1) = r[7]; M(2, 2) = r[8]; M(2, 3) = 0.0;
    M(3, 0) = 0.0; M(3, 1) = 0.0; M(3, 2) = 0.0; M(3, 3) = 1.0;
#undef M
      glTranslatef(t[0], t[1], t[2]);
      glMultMatrixf(m);
      //glPushMatrix();
      //glScalef(0.1,0.1,0.1);
      //glRotatef(180,1,0,0);
      //drawCamera();
      //glPopMatrix();
      drawFrustum(FLAGS_fov, 1.0, -0.01, -0.05);
      drawAxis(0.1);
#if 0
      glColor3f(0.3, 0.3, 0.3);
      glutSolidSphere(0.01, 20, 20); 
        glPushMatrix();
          glPushName(1);
            glColor3f(1,0,0);
            glRotatef(90,0,1,0);
            glutSolidCone(0.01, 0.1, 20, 20);
          glPopName();
        glPopMatrix();
        glPushMatrix ();
          glPushName(2);            /* Green cone is 2 */
            glColor3f(0,1,0);
            glRotatef(-90,1,0,0);
            glutSolidCone(0.01, 0.1, 20, 20);
          glPopName();
        glPopMatrix();
          glColor3f(0,0,1);         /* Blue cone is 3 */
          glPushName(3);
            glutSolidCone(0.01, 0.1, 20, 20);
          glPopName();
#endif
    glPopMatrix();
  }
}

void drawFrustum(float fovY, float aspectRatio, float nearPlane, float farPlane)
{
    float tangent = tanf(fovY/2 * DEG2RAD);
    float nearHeight = nearPlane * tangent;
    float nearWidth = nearHeight * aspectRatio;
    float farHeight = farPlane * tangent;
    float farWidth = farHeight * aspectRatio;

    // compute 8 vertices of the frustum
    float vertices[8][3];
    // near top right
    vertices[0][0] = nearWidth;     vertices[0][1] = nearHeight;    vertices[0][2] = -nearPlane;
    // near top left
    vertices[1][0] = -nearWidth;    vertices[1][1] = nearHeight;    vertices[1][2] = -nearPlane;
    // near bottom left
    vertices[2][0] = -nearWidth;    vertices[2][1] = -nearHeight;   vertices[2][2] = -nearPlane;
    // near bottom right
    vertices[3][0] = nearWidth;     vertices[3][1] = -nearHeight;   vertices[3][2] = -nearPlane;
    // far top right
    vertices[4][0] = farWidth;      vertices[4][1] = farHeight;     vertices[4][2] = -farPlane;
    // far top left
    vertices[5][0] = -farWidth;     vertices[5][1] = farHeight;     vertices[5][2] = -farPlane;
    // far bottom left
    vertices[6][0] = -farWidth;     vertices[6][1] = -farHeight;    vertices[6][2] = -farPlane;
    // far bottom right
    vertices[7][0] = farWidth;      vertices[7][1] = -farHeight;    vertices[7][2] = -farPlane;

    float colorLine1[4] = { 0.7f, 0.7f, 0.7f, 0.7f };
    float colorLine2[4] = { 0.2f, 0.2f, 0.2f, 0.7f };
    float colorPlane[4] = { 0.5f, 0.5f, 0.5f, 0.5f };

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // draw the edges around frustum
    glBegin(GL_LINES);
    glColor4fv(colorLine2);
    glVertex3f(0, 0, 0);
    glColor4fv(colorLine1);
    glVertex3fv(vertices[4]);

    glColor4fv(colorLine2);
    glVertex3f(0, 0, 0);
    glColor4fv(colorLine1);
    glVertex3fv(vertices[5]);

    glColor4fv(colorLine2);
    glVertex3f(0, 0, 0);
    glColor4fv(colorLine1);
    glVertex3fv(vertices[6]);

    glColor4fv(colorLine2);
    glVertex3f(0, 0, 0);
    glColor4fv(colorLine1);
    glVertex3fv(vertices[7]);
    glEnd();

    glColor4fv(colorLine1);
    glBegin(GL_LINE_LOOP);
    glVertex3fv(vertices[4]);
    glVertex3fv(vertices[5]);
    glVertex3fv(vertices[6]);
    glVertex3fv(vertices[7]);
    glEnd();

    glColor4fv(colorLine1);
    glBegin(GL_LINE_LOOP);
    glVertex3fv(vertices[0]);
    glVertex3fv(vertices[1]);
    glVertex3fv(vertices[2]);
    glVertex3fv(vertices[3]);
    glEnd();

    // draw near and far plane
    glColor4fv(colorPlane);
    glBegin(GL_QUADS);
    glVertex3fv(vertices[0]);
    glVertex3fv(vertices[1]);
    glVertex3fv(vertices[2]);
    glVertex3fv(vertices[3]);
    glVertex3fv(vertices[4]);
    glVertex3fv(vertices[5]);
    glVertex3fv(vertices[6]);
    glVertex3fv(vertices[7]);
    glEnd();
} 

void draw_axes(void) {
  /* Name-stack manipulation for the purpose of
     selection hit processing when mouse button
     is pressed.  Names are ignored in normal
     OpenGL rendering mode.                    */

  glPushMatrix();
                                /* No name for grey sphere */

  glColor3f(0.3,0.3,0.3);
  glutSolidSphere(0.05, 20, 20);

  glPushMatrix();
  glPushName(1);            /* Red cone is 1 */
    glColor3f(1,0,0);
    glRotatef(90,0,1,0);
    glutSolidCone(0.03, 0.3, 20, 20);
  glPopName();
  glPopMatrix();

  glPushMatrix ();
  glPushName(2);            /* Green cone is 2 */
    glColor3f(0,1,0);
    glRotatef(-90,1,0,0);
    glutSolidCone(0.03, 0.3, 20, 20);
  glPopName();
  glPopMatrix();

  glColor3f(0,0,1);         /* Blue cone is 3 */
    glPushName(3);
      glutSolidCone(0.03, 0.3, 20, 20);
    glPopName();
  glPopMatrix();
}

void load_ply(const char *filename, 
              vector <Point3f> &points, 
              vector <Color3ub> &colors,
              vector <Face> &faces){
  points.clear();
  colors.clear();
  faces.clear();

  FILE *fp = fopen(filename, "r");

  if(!fp) {
    fprintf(stderr, "Error opening %s\n", filename);
    exit(1);
  }

  char *dummy = NULL;
  char line[1024];
  char dum_s[512];
  int n_vertex;
  int n_face;

  // WARNING:
  // THIS ASSUMES A VERY STRICT FORMAT FROM MESHLAB!
  // Change this if nothing renders correctly
  dummy = fgets(line, sizeof(line), fp); 
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);
  sscanf(line, "%s %s %d", dum_s, dum_s, &n_face);
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);
  sscanf(line, "%s %s %d", dum_s, dum_s, &n_vertex);
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);
  dummy = fgets(line, sizeof(line), fp);

  points.resize(n_vertex);
  colors.resize(n_vertex);
  faces.resize(n_face);

  for(int i=0; i < n_vertex; i++) {
    dummy = fgets(line, sizeof(line), fp);

    Point3f pt;
    Color3ub col;

    sscanf(line, "%f %f %f %hhu %hhu %hhu", &pt.x, &pt.y, &pt.z, &col.r, &col.g, &col.b);

    points[i] = pt;
    colors[i] = col;
  }

  for(int i=0; i < n_face; i++) {
    int dum;
    Face face;
    dummy = fgets(line, sizeof(line), fp);
    sscanf(line, "%d %d %d %d", &dum, &face.a, &face.b, &face.c);
    faces[i] = face;
  }
}
