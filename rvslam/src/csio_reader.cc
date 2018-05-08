#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "time.h"

#include "image_file.h"
#include "image_util.h"
#include "klt_tracker.h"
#include "rvslam_util.h"
#include "visual_odometer.h"

#include "rvslam_profile.h"
rvslam::ProfileDBType pdb_;

#include "csio_reader.h"
#include "csio/csio_stream.h"
#include "csio/csio_frame_parser.h"

DEFINE_string(image_files, "test.%04d.png", "Input images.");
DEFINE_int32(reduce_size, 0, "Reduce image size.");
DEFINE_string(out, "track_%04d.png", "Tracking result images.");
DEFINE_string(pose_out, "pose.txt", "Tracking result poses.");
DEFINE_bool(egocentric, false, "Visualize the model with the camera fixed.");
DEFINE_bool(show_all_keyframes, false, "Show all keyframes from the beginning.");
DEFINE_bool(show_reprojected_pts, false, "Show reprojected 3D points.");
DEFINE_double(display_cam_size, 1.0, "Display camera axis size.");

using namespace std;
using namespace rvslam;

namespace {

template <class T> inline
string ToString(const T& v) {
  stringstream ss;
  ss << v;
  return ss.str();
}

inline float* SetPointsToBuffer(const Mat& pts, float* buf) {
  for (int j = 0; j < pts.cols(); ++j) {
    buf[0] = pts(0, j), buf[1] = pts(1, j), buf[2] = pts(2, j);
    buf += 3;
  }
  return buf;
}

Vec3 EstimateRotationalVelocity(const int idx,
                                const vector<Vec6>& pose_vec,
                                const vector<int> indices,
                                const Vec6 cur_pose) {
  Vec3 w;
  w << 0, 0, 0;
  int denom = 0;
  for (int i = 1; i < pose_vec.size(); ++i) {
    Vec6 diff = pose_vec[i] - cur_pose;
    double diff_norm = diff.norm();
    if (!diff_norm) continue;
    Vec6 rel = rvslam::VisualOdometer::RelativePoseVector(cur_pose, pose_vec[i]);
    int frmno_diff = indices[i] - idx;
    w += rel.segment(0, 3);
    denom += frmno_diff;
  }
  if (denom != 0) w /= denom;
  return w;
}

inline float* SetPointsToBuffer(const Mat& pts, const int* col_idx, int num_pts,
                                float* buf) {
  for (int j = 0; j < num_pts; ++j) {
    buf[0] = pts(0, col_idx[j]);
    buf[1] = pts(1, col_idx[j]);
    buf[2] = pts(2, col_idx[j]);
    buf += 3;
  }
  return buf;
}

void DrawGeometricOutput(const int idx, const vector<int> indices,
                         const Vec6& cur_pose,
                         const vector<Vec6>& pose_vec, const Mat3X& pts_3d,
                         vector<char>* geom_buf_ptr) {
  vector<char>& geom_buf = *geom_buf_ptr;
  static map<int, Vec6> kfposes;
  Vec6 pose = FLAGS_egocentric ? cur_pose:
      (Vec6() << 0, 0, 0, 0, 0, 0).finished();
  if (FLAGS_egocentric) {
    Mat3 R = RotationRodrigues(pose.segment(0, 3));
    pose.segment(0, 3) << 0, 0, 0;
    pose.segment(3, 3) = R.transpose() * pose.segment(3, 3);
  }

  Mat34 pose0_mat = VisualOdometer::ToPoseMatrix(pose);
  set<int> kfids;
  if (FLAGS_show_all_keyframes == false) kfposes.clear();
  for (int i = 0; i < pose_vec.size(); ++i) {
    kfposes[idx] = pose_vec[i];
    kfids.insert(idx);
  }
//kfposes.clear();  // No keyframes shown.
  const int num_kfs = kfposes.size();
  const int num_pts = pts_3d.cols();
  const int num_pred = 10;

  geom_buf.reserve(csio::ComputeGeometricObjectSize(1, num_pts, 1) +
                   csio::ComputeGeometricObjectSize(1, num_kfs * 6, 1) +
                   csio::ComputeGeometricObjectSize(2, num_pred, 1) +
                   csio::ComputeGeometricObjectSize(1, 30, 1));

  vector<csio::GeometricObject::Opt> opts(1);
  opts[0] = csio::GeometricObject::MakeOpt(
      csio::GeometricObject::OPT_POINT_SIZE, 2);
  // Plot 3D point cloud.
  LOG(INFO) << "pts: " << num_pts << " - " << geom_buf.size();
  csio::GeometricObject geom_pts =
      csio::AddGeometricObjectToBuffer('p', opts, num_pts, 1, &geom_buf);
  geom_pts.set_color(0, 0, 255, 255);
  float* pts_ptr = geom_pts.pts(0);
  for (int i = 0; i < pts_3d.cols(); ++i) {
    Vec3 p = pts_3d.col(i);
    const Vec3 pt = pose0_mat * Hom(p);
    pts_ptr = SetPointsToBuffer(pt, pts_ptr);
  }
  // Plot keyframe camera axes.
  LOG(INFO) << "kfs: " << num_kfs * 6 << " - " << geom_buf.size();
  csio::GeometricObject geom_cams =
      csio::AddGeometricObjectToBuffer('l', num_kfs * 6, 1, &geom_buf);
  geom_cams.set_color(0, 0, 0, 255);
  float* cams_ptr = geom_cams.pts(0);
  // Make a camera 3D model.
  Eigen::Matrix<double, 4, 6> cam;
  cam.fill(0.0);
  cam.row(3).fill(1.0);
  cam(0, 1) = cam(1, 3) = cam(2, 5) = FLAGS_display_cam_size;
  for (map<int, Vec6>::const_iterator it = kfposes.begin();
       it != kfposes.end(); ++it) {
    Mat34 pose_mat = VisualOdometer::ToPoseMatrix(it->second);
    Mat pt = MergedTransform(pose0_mat, InverseTransform(pose_mat)) * cam;
    cams_ptr = SetPointsToBuffer(pt, cams_ptr);
  }
  // Plot current camera pose.
  csio::GeometricObject geom_pose = csio::AddGeometricObjectToBuffer(
      'l', opts, 30, 1, &geom_buf);
  opts[0] = csio::GeometricObject::MakeOpt(
          csio::GeometricObject::OPT_LINE_WIDTH, 2);
  geom_pose.set_color(0, 255, 0, 0);
  float* pose_ptr = geom_pose.pts(0);
  {
    Mat34 pose_mat = VisualOdometer::ToPoseMatrix(cur_pose);
    Mat pt = MergedTransform(pose0_mat, InverseTransform(pose_mat)) * cam;
    pose_ptr = SetPointsToBuffer(pt, pose_ptr);
  }

  // Plot the predicted trajectory.
  opts.push_back(csio::GeometricObject::MakeOpt(
      csio::GeometricObject::OPT_LINE_STIPPLE, 0xCC));
  csio::GeometricObject pred_pts =
      csio::AddGeometricObjectToBuffer('L', opts, num_pred, 1, &geom_buf);
  pred_pts.set_color(0, 255, 165, 0);
  float* pred_ptr = pred_pts.pts(0);
  if (FLAGS_show_all_keyframes == false) {
    Mat34 pose_mat = VisualOdometer::ToPoseMatrix(cur_pose);
    Mat pt = MergedTransform(pose0_mat, InverseTransform(pose_mat)) * cam;
    Mat3 R0 = RotationRodrigues(cur_pose.segment(0, 3)).transpose();
    Vec3 w = EstimateRotationalVelocity(idx, pose_vec, indices, cur_pose);
    w(0) = w(2) = 0;
    Mat3 R = RotationRodrigues(w * 10).transpose();
    Vec3 unit, pos;
    pos << 0, 0, 0;
    unit << 0, 0, 0.1;
    pred_ptr = SetPointsToBuffer(pos, pred_ptr);
    for (int i = 1; i < num_pred; ++i) {
      unit = R * unit;
      pos += unit * 10;
      pred_ptr = SetPointsToBuffer(R0 * pos, pred_ptr);
    }
  }
}

}  // namespace



// idx : image file num
// image_pts : feature location(x,y)
// cur_pose : current keyframe's pose
// pose_vec : all keyframes pose vec
// 3d_pts : landmarks
// indices : keyframes indices(idx of keyframe vector)
void Csio::Process(const int idx, const Mat3X& image_pts, 
                   const Vec6& cur_pose, const vector<Vec6>& pose_vec,
                   const vector<int> indices, const Mat3X& pts_3d) {

  //Visualization
  if (FLAGS_pose_out.empty() == false) {
    ofs_pose_out.open(FLAGS_pose_out.c_str(), fstream::out);
  }

  // Load the images.
  // Set image filename
  MCImageRGB8 image;
  MCImageGray8 image_gray;
  const string image_path= StringPrintf(FLAGS_image_files.c_str(), idx);
  ReadImageRGB8(image_path, &image);
  RGB8ToGray8(image, &image_gray);
  MCImageGray8::ArrayType image_array = image.GetPlane();
  for (int r = 0; r < FLAGS_reduce_size; ++r) {
    MCImageGray8::ArrayType reduced_array;
    ReduceSize(image_array, &reduced_array);
    image_array.swap(reduced_array);
  }
  LOG(INFO) << "Frame " << idx << ": "
    << image_array.rows() << "x" << image_array.cols();

  //Visualization
  // Setup CSIO output stream if not done.
  if (idx == 0 && FLAGS_out.empty() == false) {
    vector<csio::ChannelInfo> channels;
    const int w = image_array.rows(), h = image_array.cols();
    channels.push_back(csio::ChannelInfo(
          0, csio::MakeImageTypeStr("rgb8", w, h), "output"));
    channels.push_back(csio::ChannelInfo(
          1, csio::MakeGeometricObjectTypeStr(w, h), "output"));
    map<string, string> config;
    if (csio_out.Setup(channels, config, FLAGS_out) == true) {
      LOG(INFO) << "csio::OutputStream opened (out=" << FLAGS_out << ").";
    }
  }

  if (ofs_pose_out.is_open()) {
    //Set Pose
    const Vec6 pose = cur_pose;
    Vec6 relative_pose = VisualOdometer::RelativePoseVector(prev_pose, pose);
    ofs_pose_out << idx << " " << pose.transpose() << " \t"
      << relative_pose.transpose()
      << "\n";
    prev_pose = pose;
  }

  if (csio_out.IsOpen()) {
    MCImageGray8::ArrayType out_array;
    //ReduceSize(image_array, &out_array);
    const double s = 1;
    MCImageRGB8 out;
    out.SetAllPlanes(image_array);
    const int n = image_pts.cols();

    if (!FLAGS_show_reprojected_pts) {
      for (int i = 0; i < n; ++i) {
        const int x = image_pts(0,i);
        const int y = image_pts(1,i);
        const MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(0, 0, 255);
        DrawDot(out, s * x, s * y, c);
      }
    }
    DrawTextFormat(out, 5, 5, MakePixelRGB8(255, 255, 255), "%03d", idx);
    LOG(INFO) << "rgb8 " <<  image.width() << ", " <<  image.height()
      << " - " << out.size();

    vector<char> geom_buf;
    DrawGeometricOutput(idx, indices, cur_pose, pose_vec, pts_3d, &geom_buf);

    csio_out.PushSyncMark(2);
    csio_out.Push(0, out.data(), out.size() * 3);
    csio_out.Push(1, geom_buf.data(), geom_buf.size());
  }
}

