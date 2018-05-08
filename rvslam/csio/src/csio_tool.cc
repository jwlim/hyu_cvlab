// csio_tool.cc
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include "csio_stream.h"

#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <glog/logging.h>
#include <gflags/gflags.h>

using namespace std;

DEFINE_string(in, "-", "File path for csio::InputStream (- for stdin).");
DEFINE_string(out, "", "File path for csio::OutputStream (- for stdout).");
DEFINE_string(cmd, "", "Command.");
DEFINE_int32(skip, 0, "Skip the given number of frames.");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Setup csio::InputStream.
  LOG(INFO) << "setting up csio::Inputstream (in=" << FLAGS_in << ").";
  csio::InputStream csio_is;
  if (csio_is.Setup(FLAGS_in) == false) {
    LOG(ERROR) << "failed to open csio::InputStream (in=" << FLAGS_in << ").";
    return -1;
  }
  LOG(INFO) << "setup csio::InputStream(" << FLAGS_in << ") complete.";

  // Setup csio::OutputStream
  LOG(INFO) << "setting up csio::OutputStream (out=" << FLAGS_out << ").";
  csio::OutputStream csio_os;
  map<string, string> config;
  if (csio_os.Setup(csio_is.channels(), config, FLAGS_out) == false) {
    LOG(ERROR) << "failed to open csio::OutputStream, out=" << FLAGS_out;
    return -1;
  }
  LOG(INFO) << "setup csio::OutputStream (out=" << FLAGS_out << ") complete.";

  // Copy the stream.
  csio::Frame frame;
  for (int cnt = 0; csio_is.Fetch(&frame); ++cnt) {
    if (cnt < FLAGS_skip) continue;
    LOG_EVERY_N(INFO, 50) << "dumping frame " << cnt;
    csio_os.Push(frame);
  }
  LOG(INFO) << "finished dumping...";

  csio_os.Close();
  csio_is.Close();

  return 0;
}
