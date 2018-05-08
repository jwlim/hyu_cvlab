#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "csio_channel.h"
#include "csio_stream.h"
#include "csio_util.h"
#include "csio_frame_parser.h"

using namespace std;

DEFINE_string(in, "-", "File path for csio::InputStream (- for stdin).");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  
  LOG(INFO) << "setting up csio::Inputstream (in=" << FLAGS_in << ").";
  csio::InputStream csin;
  if (csin.Setup(FLAGS_in) == false) {
    LOG(ERROR) << "failed to open csio::InputStream (in=" << FLAGS_in << ").";
    return -1;
  }
  LOG(INFO) << "setup csio::InputStream(" << FLAGS_in << ") complete.";

  const vector<csio::ChannelInfo>& channels = csin.channels();
  if (channels.size() != 1) {
    LOG(ERROR) << "imu reader only reads 1 channel.";
    return -1;
  }
  
  const csio::ChannelInfo& ch_info = channels[0];
  map <string, string> cfg;
  string type = csio::ParseTypeStr(ch_info.type, &cfg);

  LOG(INFO) << "csio_type '" << ch_info.type << "' -> '" << type << "'";
  for (map<string, string>::const_iterator i = cfg.begin(); i != cfg.end(); ++i)
    LOG(INFO) << "cfg: '" << i->first << "' : '" << i->second << "'";

  if (!csio::IsIMUType(type, cfg)) {
    LOG(ERROR) << "unknown imu_type '" << type << "'.";
    return -1;
  }
  
  vector<csio::Frame> frame_array_ptr;

  while (true) {
    if (csin.FetchSyncFrames(&frame_array_ptr) == false) {
      break;    
    }
    csio::Frame frame = frame_array_ptr.back();
    printf("%70s\r", frame.buf.data());
  }
}

