// feat_tracker_main.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include <iostream>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "image_file.h"
#include "image_util.h"
#include "feat_opencv.h"

#include "rvslam_profile.h"
rvslam::ProfileDBType pdb_;

#include "csio/csio_stream.h"
#include "csio/csio_frame_parser.h"

using namespace std;
using namespace rvslam;

DEFINE_string(image, "test_klt_%04d_0.png", "Input images.");
DEFINE_string(out, "track_%04d.png", "Tracking result images.");
DEFINE_int32(start, 0, "Start index of files to process.");
DEFINE_int32(end, 1, "End index of files to process.");
DEFINE_int32(step, 1, "Step in index of files to process.");
DECLARE_int32(klt_num_levels);

DEFINE_string(feature, "FAST", "feature : FAST, ORB ...");

namespace {

template <class T> inline
string ToString(const T& v) {
  stringstream ss;
  ss << v;
  return ss.str();
}

}  // namespace

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  FeatureDetector tracker;
  vector<int> prev_ftids;
  vector<cv::KeyPoint> prev_keypoints;
  // TODO: reduce_size is not implemented correctly.

  MCImageRGB8 image_rgb;
  MCImageGray8 image;
  csio::OutputStream csio_out;
  for (int idx = FLAGS_start; idx <= FLAGS_end; idx += FLAGS_step) {
ProfileBegin("-.frate", &pdb_);
ProfileBegin("0.load ", &pdb_);
    const string image_path = StringPrintf(FLAGS_image.c_str(), idx);
    if (!ReadImageRGB8(image_path, &image_rgb)) break;
    RGB8ToGray8(image_rgb, &image);
    MCImageGray8::ArrayType array = image.GetPlane();
    LOG(INFO) << "Frame " << idx << ": " << array.rows() << "x" << array.cols();
ProfileEnd("0.load ", &pdb_);

ProfileBegin("1.track", &pdb_);
    if (idx == FLAGS_start) {
      tracker.Setup(FLAGS_feature, FLAGS_image, idx, 0);
    } else {
      tracker.Process(FLAGS_feature, FLAGS_image, idx, idx - 1,
                      0, tracker.last_ftid());
    }
ProfileEnd("1.track", &pdb_);

ProfileBegin("2.out  ", &pdb_);
    const vector<int>& ftids = tracker.ftids(); 
    const vector<cv::KeyPoint>& keypoints = tracker.keypoints();
//    const vector<KLTFeat>& feats = tracker.features();
    map<int, int> idmap;
    for (int i = 0; i < prev_ftids.size(); ++i) idmap[prev_ftids[i]] = i;

    MCImageRGB8 out;
    out.SetAllPlanes(array);
LOG(INFO) << array.rows() << "," << array.cols();
//    image.SetPlane(0, array);
//    Gray8ToRGB8(image, &out);
    for (int i = 0; i < keypoints.size(); ++i) {
      const int ftid = ftids[i];
      const cv::KeyPoint& kp = keypoints[i];
      bool prev_ft_found = (idmap.count(ftid) > 0);
      if (prev_ft_found) {
        const cv::Point2f& p = prev_keypoints[idmap[ftid]].pt;
        const MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(0, 0, 255);
        DrawLine(out, p.x, p.y, kp.pt.x, kp.pt.y, c);
        DrawDot(out, p.x, p.y, c);
      }
      DrawDot(out, kp.pt.x, kp.pt.y,
              MCImageRGB8::MakePixel(prev_ft_found ? 255 : 0, 255, 0));
    }
    if (FLAGS_out == "-") {  // Use CSIO streaming.
      if (idx == FLAGS_start) {
        vector<csio::ChannelInfo> channels;
        const int w = out.width(), h = out.height();
        channels.push_back(csio::ChannelInfo(
            0, csio::MakeImageTypeStr("rgb8", w, h), "output"));
        map<string, string> config;
        if (csio_out.Setup(channels, config, FLAGS_out) == true) {
          LOG(INFO) << "csio::OutputStream opened (out=" << FLAGS_out << ").";
        }
      }
      if (csio_out.IsOpen()) csio_out.Push(0, out.data(), out.size() * 3);
    } else {
      WriteImageRGB8(out, StringPrintf(FLAGS_out.c_str(), idx));
    }
ProfileEnd("2.out  ", &pdb_);

    prev_ftids = ftids;
    prev_keypoints = keypoints;
ProfileEnd("-.frate", &pdb_);
ProfileDump(pdb_);
  }
//  tracker.Cleanup();
  if (csio_out.IsOpen()) csio_out.Close();
  return 0;
}

