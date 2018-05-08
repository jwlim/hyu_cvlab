#include "csiomod_camera1394.h"
#include "csio_stream.h"
#include "csio_frame_parser.h"

#include <fstream>
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace std;

DEFINE_string(out, "-", "File path for csio::OutputStream (- for stdout).");
DEFINE_bool(listdev, false, "Print list of available devices.");
DEFINE_int32(devno, 0, "Device number to use.");
DEFINE_string(modestr, "640x480.rgb8", 
  "Device mode (\"640x480.rgb8\" or \"fmt7.1.rgb8.644.482.0.0\")");
DEFINE_string(fratestr, "modeset", "Frame rate for camera.");

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  
  dc1394error_t err;
  dc1394_t *d = dc1394_new ();
  if (!d)
    return 1;

  if (FLAGS_listdev) {
    ListDev(d);
    return 0;
  } 

  Camera1394 cam1394 = Camera1394(GetDevice(d, FLAGS_devno));
  LOG(INFO) << "Camera vendor: '" << cam1394.cam->vendor << "', model: '"
    << cam1394.cam->model << "'";

  if (!cam1394.Setup(FLAGS_modestr, FLAGS_fratestr)) {
    dc1394_free(d);
    return 1;
  }
  LOG(INFO) << "Successfully setup the camera";
  if (cam1394.IsFmt7()) {
    LOG(INFO) << "mode='Format7." << cam1394.mode - DC1394_VIDEO_MODE_FORMAT7_0
      << "', coding='" << cam1394.codingstr 
      << "', size=" << cam1394.w << "x" << cam1394.h
      << "." << cam1394.x << "@" << cam1394.y;
  } else {
    LOG(INFO) << "mode='" << FLAGS_modestr 
      << "', frame rate='" << FLAGS_fratestr << "'";  
  }

  vector<csio::ChannelInfo> channels;
  channels.push_back(csio::ChannelInfo(
    0, csio::MakeImageTypeStr(cam1394.codingstr, cam1394.w, cam1394.h), "1394"));
  map<string, string> config;

  csio::OutputStream csio_os;
  if (csio_os.Setup(channels, config, FLAGS_out)) {
    LOG(INFO) << "csio::OutputStream opened (out=" << FLAGS_out << "),"
      << "w:" << cam1394.w << "h:" << cam1394.h;
  } else {
    dc1394_free(d);
    return 1;
  }

  if (!cam1394.StartCapture()) {
    dc1394_free(d);
    return 1;
  }
  LOG(INFO) << "Start capturing";

  dc1394video_frame_t *frame;
  bool die(false);
  while(!die){
    if (!cam1394.FetchFrame(&frame)) {
      dc1394_free(d);
      return 1;
    } 
    csio_os.Push(0, frame->image , frame->image_bytes);
    if (!cam1394.DropFrame(&frame)) {
      dc1394_free(d);
      return 1;
    }
  }

  if (!cam1394.StopCapture()) {
    dc1394_free(d);
    return 1;
  }
  LOG(INFO) << "Stop capturing";
  
  dc1394_capture_stop(cam1394.cam);
  dc1394_camera_free(cam1394.cam);
  dc1394_free (d);

  return 0;
}
