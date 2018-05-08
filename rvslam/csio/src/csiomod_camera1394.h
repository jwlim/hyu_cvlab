#ifndef _CSIOMOD_CAMERA1394_H
#define _CSIOMOD_CAMERA1394_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <iostream>
#include <string.h>
#include <string>

#include <dc1394/dc1394.h>

using namespace std;

// careful! these correspond to #define'd features - order matters!
static const char *g_featurenames[] = {
  "brightness", "exposure", "sharpness", "white_balance", "hue", "saturation",
  "gamma", "shutter", "gain", "iris", "focus", "temperature", "trigger",
  "trigger_delay", "white_shading", "frame_rate", "zoom", "pan", "tilt",
  "optical_filter", "capture_size", "capture_quality", NULL };

static const char *g_colorcodingnames[] = {
  "mono8", "yuv411", "yuv422", "yuv444", "rgb8", "mono16", "rgb16", "mono16s",
  "rgb16s", "raw8", "raw16", NULL };

class Camera1394{
public:
  dc1394camera_t *cam;
  dc1394video_mode_t mode;
  dc1394framerate_t frate;

  /* format 7 */
  dc1394format7mode_t f7;
  dc1394color_coding_t color_coding;
  string codingstr;
  int x, y, w, h;

  Camera1394(dc1394camera_t *cam);
  ~Camera1394();

  void ListAvailableFeatures();
  void ListSupportedModes();
  bool ParseMode(string modestr, string fratestr);
  bool ParseFmt7Mode(string modestr);
  bool Setup(string modestr, string fratestr);
  bool StartCapture();
  bool StopCapture();
  bool FetchFrame(dc1394video_frame_t **frame);
  bool DropFrame(dc1394video_frame_t **frame);

  bool IsFmt7();
};

dc1394camera_t* GetDevice(dc1394_t *d, int32_t devno);
void ListDev(dc1394_t *d);

inline int _toi(string str) {
  return atoi(str.c_str());
}

#endif
