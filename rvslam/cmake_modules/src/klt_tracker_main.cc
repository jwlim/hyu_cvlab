// klt_tracker_main.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include <iostream>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "image_file.h"
#include "klt_tracker.h"

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
DEFINE_int32(reduce_size, 0, "Reduce image size.");
DECLARE_int32(klt_num_levels);

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

  KLTTracker tracker;
  MCImageRGB8 image_rgb;
  MCImageGray8 image;
  vector<KLTFeat> prev_feats;
  csio::OutputStream csio_out;
  for (int idx = FLAGS_start; idx <= FLAGS_end; idx += FLAGS_step) {
ProfileBegin("-.frate", &pdb_);
ProfileBegin("0.load ", &pdb_);
    const string image_path = StringPrintf(FLAGS_image.c_str(), idx);
    if (!ReadImageRGB8(image_path, &image_rgb)) break;
    RGB8ToGray8(image_rgb, &image);
    MCImageGray8::ArrayType array = image.GetPlane();
    for (int r = 0; r < FLAGS_reduce_size; ++r) {
      MCImageGray8::ArrayType reduced_array;
      ReduceSize(array, &reduced_array);
      array.swap(reduced_array);
    }
    LOG(INFO) << "Frame " << idx << ": " << array.rows() << "x" << array.cols();
ProfileEnd("0.load ", &pdb_);

ProfileBegin("1.track", &pdb_);
    if (idx == FLAGS_start) {
      tracker.Setup(array);
    } else {
      tracker.Process(array);
    }
ProfileEnd("1.track", &pdb_);

ProfileBegin("2.out  ", &pdb_);
    const vector<KLTFeat>& feats = tracker.features();
    map<int, int> idmap;
    for (int i = 0; i < prev_feats.size(); ++i) idmap[prev_feats[i].id] = i;

    MCImageRGB8 out;
    out.SetAllPlanes(array);
LOG(INFO) << array.rows() << "," << array.cols();
//    image.SetPlane(0, array);
//    Gray8ToRGB8(image, &out);
    for (int i = 0; i < feats.size(); ++i) {
      const KLTFeat& ft = feats[i];
      bool prev_ft_found = (idmap.count(ft.id) > 0);
      if (prev_ft_found) {
        const Eigen::Vector2f& p = prev_feats[idmap[ft.id]].pos;
        const MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(0, 0, 255);
        DrawLine(out, p(0), p(1), ft.pos(0), ft.pos(1), c);
        DrawDot(out, p(0), p(1), c);
      }
      DrawDot(out, ft.pos(0), ft.pos(1),
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

    prev_feats = feats;
ProfileEnd("-.frate", &pdb_);
ProfileDump(pdb_);
  }
  tracker.Cleanup();
  if (csio_out.IsOpen()) csio_out.Close();
  return 0;
}

