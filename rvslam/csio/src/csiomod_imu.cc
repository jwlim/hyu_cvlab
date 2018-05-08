#include "csio_stream.h"
#include "csio_frame_parser.h"

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

  map<string, string> cfg;
  vector<csio::ChannelInfo> channels;
  channels.push_back(csio::ChannelInfo(
          0, csio::MakeIMUTypeStr(), "imu"));

  csio::OutputStream csio_os;
  if (csio_os.Setup(channels, cfg, FLAGS_out) == true) {
    LOG(INFO) << "csio::OutputStream opened (out=" << FLAGS_out << ")";
  } else {
    return -1;
  }

  int n = 0, i = 0;
  char buf[512];
  const char SOL = '*', EOL = '\n';
  bool incoming(false), die(false);

  while (!die) {
    n = read(fd_imu, &buf[i], 1);
    if (n > 0) {
      if (!incoming && buf[i] == SOL) {
        incoming = true;
      } else if (incoming) {
        if (buf[i] == EOL) {
          buf[i] = '\0';
          csio_os.Push(0, buf , i);
          i = 0;
          incoming = false;
        } else {
          i++;
        }
      }
    }
  }

  close(fd_imu);

  return 0;
}
