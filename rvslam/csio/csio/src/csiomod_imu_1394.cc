#include "csio_stream.h"
#include "csio_frame_parser.h"
#include "csiomod_camera1394.h"

#include <fstream>
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <string.h>     // string function definitions
#include <unistd.h>     // UNIX standard function definitions
#include <fcntl.h>      // File control definitions
#include <errno.h>      // Error number definitions
#include <termios.h>    // POSIX terminal control definitions
#include <stropts.h>
using namespace std;

DEFINE_string(out, "-", "File path for csio::OutputStream (- for stdout).");
DEFINE_string(port, "ttyUSB0", "Serial port name for IMU sensor.");
DEFINE_bool(accelero, false, "Set output accelero.");
DEFINE_bool(magneto, false, "Set output magneto.");
DEFINE_bool(listdev, false, "Print list of available devices.");
DEFINE_int32(devno, 0, "Device number to use.");
DEFINE_string(modestr, "640x480.rgb8", 
  "Device mode (\"640x480.rgb8\" or \"fmt7.1.rgb8.644.482.0.0\")");
DEFINE_string(fratestr, "modeset", "Frame rate for camera.");

int main (int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  string port_name = string("/dev/").append(FLAGS_port);

  /* Open File Descriptor */
  int fd_imu = open(port_name.c_str(), O_RDWR| O_NOCTTY | O_NONBLOCK | O_NDELAY );
  if (fd_imu < 0) {
    LOG(ERROR) << "Error(" << errno << ") opening " << port_name << ": " <<
      strerror (errno);
    return -1;
  }

  struct termios tty;
  struct termios tty_old;
  memset (&tty, 0, sizeof tty);

  /* Get currpnt attributes of tty */
  if (tcgetattr(fd_imu, &tty) != 0 ) {
    LOG(ERROR) << "Error(" << errno << ") from tcgetattr: " <<
      strerror(errno);
    return -1;
  }

  /* Save old tty parameters */
  tty_old = tty;

  /* Set Baud Rate */
  cfsetospeed (&tty, (speed_t)B115200);
  cfsetispeed (&tty, (speed_t)B115200);

  /* Setting other Port Stuff */
  tty.c_cflag &= ~PARENB;            // Make 8n1
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;
  tty.c_cflag &= ~CRTSCTS;           // no flow control
  tty.c_cc[VMIN] = 0;                // read doesn't block
  tty.c_cc[VTIME] = 5;               // 0.5 seconds read timeout
  tty.c_cflag |= CREAD | CLOCAL;     // turn on READ & ignore ctrl lines

  /* Set new attributes */
  if (tcsetattr(fd_imu, TCSANOW, &tty) != 0) {
    LOG(ERROR) << "Error(" << errno << ") from tcsetattr: " <<
      strerror(errno);;
  }

  /* Set output mode */
  if (FLAGS_accelero) write(fd_imu, "<soa1>", 6);
  if (FLAGS_magneto) write(fd_imu, "<som1>", 6);

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


  map<string, string> cfg;
  vector<csio::ChannelInfo> channels;
  channels.push_back(csio::ChannelInfo(
    0, csio::MakeImageTypeStr(cam1394.codingstr, cam1394.w, cam1394.h), "1394"));
  channels.push_back(csio::ChannelInfo(
          1, csio::MakeIMUTypeStr(), "imu"));

  csio::OutputStream csio_os;
  if (csio_os.Setup(channels, cfg, FLAGS_out) == true) {
    LOG(INFO) << "csio::OutputStream opened (out=" << FLAGS_out << ")";
  } else {
    return -1;
  }

  if (!cam1394.StartCapture()) {
    dc1394_free(d);
    return 1;
  }
  LOG(INFO) << "Start capturing";

  int n = 0, i = 0;
  char buf[512];
  const char SOL = '*', EOL = '\n';
  bool incoming(false), die(false);
  dc1394video_frame_t *frame;

  while (!die) {
    n = read(fd_imu, &buf[i], 1);
    if (n > 0) {
      if (!incoming && buf[i] == SOL) {
        incoming = true;
      } else if (incoming) {
        if (buf[i] == EOL) {
          buf[i] = '\0';
          if (!cam1394.FetchFrame(&frame)) {
            dc1394_free(d);
            return 1;
          }
          csio_os.PushSyncMark(2);
          csio_os.Push(0, frame->image , frame->image_bytes);
          csio_os.Push(1, buf , i);
          if (!cam1394.DropFrame(&frame)) {
            dc1394_free(d);
            return 1;
          }
          i = 0;
          incoming = false;
        } else {
          i++;
        }
      }
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
  close(fd_imu);

  return 0;
}
