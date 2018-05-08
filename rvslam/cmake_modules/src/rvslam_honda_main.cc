#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "image_file.h"
#include "image_util.h"
#include "klt_tracker.h"
#include "rvslam_util.h"
#include "visual_odometer.h"
#include "feature.h"

#include "rvslam_profile.h"
rvslam::ProfileDBType pdb_;

#include "csio/csio_stream.h"
#include "csio/csio_frame_parser.h"

using namespace std;
using namespace rvslam;

DEFINE_string(stereo_calib, "",
              "Comma-separated stereo calibration (fx,fy,cx,cy,baseline).");
DEFINE_string(left, "test_klt_%04d_0.png", "Input images.");
DEFINE_string(right, "test_klt_%04d_1.png", "Input images.");
DEFINE_string(out, "track_%04d.png", "Tracking result images.");
DEFINE_int32(start, 0, "Start index of files to process.");
DEFINE_int32(end, 1, "End index of files to process.");
DEFINE_int32(step, 1, "Step in index of files to process.");
DEFINE_int32(reduce_size, 0, "Reduce image size.");
DEFINE_int32(max_disparity, 0, "Max disparity value.");

DEFINE_string(pose_out, "pose.txt", "Tracking result poses.");
DEFINE_bool(egocentric, false, "Visualize the model with the camera fixed.");
DEFINE_bool(show_all_keyframes, false, "Show all keyframes from the beginning.");
DEFINE_bool(show_reprojected_pts, false, "Show reprojected 3D points.");
DEFINE_double(display_cam_size, 1.0, "Display camera axis size.");

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

void DrawGeometricOutput(const VisualOdometer& vo,
                         vector<char>* geom_buf_ptr) {
  vector<char>& geom_buf = *geom_buf_ptr;
  static map<int, Vec6> kfposes;

  Vec6 pose = FLAGS_egocentric ? vo.pose() :
      (Vec6() << 0, 0, 0, 0, 0, 0).finished();
  if (FLAGS_egocentric) {
    Mat3 R = RotationRodrigues(pose.segment(0, 3));
    pose.segment(0, 3) << 0, 0, 0;
    pose.segment(3, 3) = R.transpose() * pose.segment(3, 3);
  }

  Mat34 pose0_mat = VisualOdometer::ToPoseMatrix(pose);
  const vector<VisualOdometer::Keyframe>& keyframes = vo.keyframes();
  const map<int, Vec3>& ftid_pts_map = vo.ftid_pts_map();
  set<int> kfids;
  if (FLAGS_show_all_keyframes == false) kfposes.clear();
  for (int i = 0; i < keyframes.size(); ++i) {
    const VisualOdometer::Keyframe& kf = keyframes[i];
    kfposes[kf.frmno] = kf.pose;
    kfids.insert(kf.frmno);
  }
//kfposes.clear();  // No keyframes shown.
  const int num_kfs = kfposes.size();
  const int num_pts = ftid_pts_map.size();
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
  for (map<int, Vec3>::const_iterator it = ftid_pts_map.begin();
       it != ftid_pts_map.end(); ++it) {
    const Vec3 pt = pose0_mat * Hom(it->second);
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
    Mat34 pose_mat = VisualOdometer::ToPoseMatrix(vo.pose());
    Mat pt = MergedTransform(pose0_mat, InverseTransform(pose_mat)) * cam;
    pose_ptr = SetPointsToBuffer(pt, pose_ptr);
  }
/*
  if (FLAGS_show_all_keyframes == false) {
    Eigen::Matrix<double, 4, 8> car;
    car << -1, 1, 1, -1, -1, 1, 1, -1,
            0, 0, 0, 0, 1, 1, 1, 1,
            -2, -2, 3, 3, -2, -2, 3, 3,
            1, 1, 1, 1, 1, 1, 1, 1;
    car.block(0, 0, 3, 8) *= 0.4;
    const int idx[24] = { 0,1, 1,2, 2,3, 3,0,  4,5, 5,6, 6,7, 7,4,
                          0,4, 1,5, 2,6, 3,7 };
    Mat34 pose_mat = VisualOdometer::ToPoseMatrix(vo.pose());
//    Mat pt = InverseTransform(pose_mat) * cam;
    Mat pt = MergedTransform(pose0_mat, InverseTransform(pose_mat)) * car;
    pose_ptr = SetPointsToBuffer(pt, idx, 24, pose_ptr);
  }
*/
  // Plot the predicted trajectory.
  opts.push_back(csio::GeometricObject::MakeOpt(
      csio::GeometricObject::OPT_LINE_STIPPLE, 0xCC));
  csio::GeometricObject pred_pts =
      csio::AddGeometricObjectToBuffer('L', opts, num_pred, 1, &geom_buf);
  pred_pts.set_color(0, 255, 165, 0);
  float* pred_ptr = pred_pts.pts(0);
  if (FLAGS_show_all_keyframes == false) {
    Mat34 pose_mat = VisualOdometer::ToPoseMatrix(vo.pose());
    Mat pt = MergedTransform(pose0_mat, InverseTransform(pose_mat)) * cam;
    Mat3 R0 = RotationRodrigues(vo.pose().segment(0, 3)).transpose();
    Vec3 w = vo.EstimateRotationalVelocity();
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

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_stereo_calib.empty()) {
    LOG(ERROR) << "stereo_calib is not given (fx,fy,cx,cy,baseline)";
    return -1;
  }
  VisualOdometer::StereoCalib calib;
  CHECK_EQ(5, sscanf(&FLAGS_stereo_calib[0], "%lf,%lf,%lf,%lf,%lf",
        &calib.fx, &calib.fy, &calib.cx, &calib.cy, &calib.baseline))
      << FLAGS_stereo_calib;

  VisualOdometer vo(calib);
  StereoKLTTracker tracker;
  MCImageRGB8 left_rgb, right_rgb;
  MCImageGray8 left, right;
  vector<KLTFeat> prev_left_feats, prev_right_feats;

  csio::OutputStream csio_out;
  ofstream ofs_pose_out;
  if (FLAGS_pose_out.empty() == false) {
    ofs_pose_out.open(FLAGS_pose_out.c_str(), fstream::out);
  }

  vector<int> prev_ftids;
  vector<KLTFeat> prev_feats;
  map<int, Vec6> kfposes;
  Vec6 prev_pose;
  prev_pose << 0, 0, 0, 0, 0, 0;
  for (int idx = FLAGS_start; idx <= FLAGS_end; idx += FLAGS_step) {
ProfileBegin("-.frate", &pdb_);
ProfileBegin("0.load ", &pdb_);
    // Load the left and right images.
    const string left_path = StringPrintf(FLAGS_left.c_str(), idx);
    const string right_path = StringPrintf(FLAGS_right.c_str(), idx);
    if (!ReadImageRGB8(left_path, &left_rgb)) break;
    if (!ReadImageRGB8(right_path, &right_rgb)) break;
    RGB8ToGray8(left_rgb, &left);
    RGB8ToGray8(right_rgb, &right);
    MCImageGray8::ArrayType left_array = left.GetPlane();
    MCImageGray8::ArrayType right_array = right.GetPlane();
    for (int r = 0; r < FLAGS_reduce_size; ++r) {
      MCImageGray8::ArrayType reduced_array;
      ReduceSize(left_array, &reduced_array);
      left_array.swap(reduced_array);
      ReduceSize(right_array, &reduced_array);
      right_array.swap(reduced_array);
    }
    LOG(INFO) << "Frame " << idx << ": "
        << left_array.rows() << "x" << left_array.cols();
ProfileEnd("0.load ", &pdb_);

ProfileBegin("1.track", &pdb_);
    if (idx == FLAGS_start) {
      tracker.Setup(left_array, right_array);
    } else {
      tracker.Process(left_array, right_array);
    }
ProfileEnd("1.track", &pdb_);

    // Setup CSIO output stream if not done.
    if (idx == FLAGS_start && FLAGS_out.empty() == false) {
      vector<csio::ChannelInfo> channels;
      const int w = left_array.rows() / 2, h = left_array.cols() / 2;
      channels.push_back(csio::ChannelInfo(
          0, csio::MakeImageTypeStr("rgb8", w, h), "output"));
      channels.push_back(csio::ChannelInfo(
          1, csio::MakeGeometricObjectTypeStr(w, h), "output"));
      map<string, string> config;
      if (csio_out.Setup(channels, config, FLAGS_out) == true) {
        LOG(INFO) << "csio::OutputStream opened (out=" << FLAGS_out << ").";
      }
    }

ProfileBegin("2.vodo ", &pdb_);
    const vector<KLTFeat>& left_feats = tracker.left().features();
    const vector<KLTFeat>& right_feats = tracker.right().features();
    typedef StereoKLTTracker::IDPair IDPair;
    const vector<IDPair>& matches = tracker.matched_features();

    vector<int> ftids(matches.size());
    vector<KLTFeat> feats(matches.size());
    Mat3X image_pts(3, matches.size());
    for (int i = 0; i < matches.size(); ++i) {
//      if (i > 0) CHECK_GT(feats[i].id, feats[i - 1].id);
      const KLTFeat& ft = (feats[i] = left_feats[matches[i].first]);
      const KLTFeat& ft_right = right_feats[matches[i].second];
      const double x = ft.pos(0), y = ft.pos(1);
      const double disparity = x - ft_right.pos(0);
      ftids[i] = ft.id;
      image_pts.col(i) << x, y, disparity;
    }
//LOG(INFO) << "max: " << setfill(' ') << image_pts.rowwise().minCoeff().transpose();
    vo.ProcessFrame(idx, ftids, image_pts);
ProfileEnd("2.vodo ", &pdb_);

    if (ofs_pose_out.is_open()) {
ProfileBegin("3.out", &pdb_);
      const Vec6 pose = vo.pose();
      Vec6 relative_pose = VisualOdometer::RelativePoseVector(prev_pose, pose);
      ofs_pose_out << idx << " " << pose.transpose() << " \t"
          << relative_pose.transpose()
//          << " \t" << vo.EstimateRotationalVelocity().transpose()
          << "\n";
      prev_pose = pose;
    }

    if (csio_out.IsOpen()) {
      MCImageGray8::ArrayType out_array;
      ReduceSize(left_array, &out_array);
      const double s = 0.5;
      MCImageRGB8 out;
//      out.SetAllPlanes(left_array);
      out.SetAllPlanes(out_array);
      const vector<int>& pose_inliers = vo.pose_inliers();

      Mat2X proj_pos;
      const Mat2X& proj_pos_veh = vo.veh_fpos;
      const Mat2X& proj_pos_p3p = vo.p3p_fpos;
      vo.GetFeatureProjections(ftids, vo.pose(), &proj_pos);
      for (int i = 0; i < feats.size(); ++i) {
        const KLTFeat& ft = feats[i];
        int prev_idx = FindInSortedArray(prev_ftids, ft.id);
        if (FLAGS_show_reprojected_pts) {
          if (i < proj_pos_veh.cols() &&
              proj_pos_veh(0, i) >= 0 && proj_pos_veh(1, i) >= 0) {
            Eigen::Vector2f p = proj_pos_veh.col(i).cast<float>();
            MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(128, 0, 255);
            DrawLine(out, s * p[0], s * p[1], s * ft.pos(0), s * ft.pos(1), c);
            DrawDot(out, s * p[0], s * p[1], c);
          }
          if (i < proj_pos_p3p.cols() &&
              proj_pos_p3p(0, i) >= 0 && proj_pos_p3p(1, i) >= 0) {
            Eigen::Vector2f p = proj_pos_p3p.col(i).cast<float>();
            MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(0, 0, 255);
            DrawLine(out, s * p[0], s * p[1], s * ft.pos(0), s * ft.pos(1), c);
            DrawDot(out, s * p[0], s * p[1], c);
          }
//          if (proj_pos(0, i) >= 0 && proj_pos(1, i) >= 0) {
//            Eigen::Vector2f p = proj_pos.col(i).cast<float>();
//            MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(0, 0, 255);
//            DrawLine(out, p[0], p[1], ft.pos(0), ft.pos(1), c);
//            DrawDot(out, p[0], p[1], c);
//          }
        }
        bool inlier = binary_search(pose_inliers.begin(), pose_inliers.end(),
                                    ft.id);
        DrawDot(out, s * ft.pos(0), s * ft.pos(1),
                MakePixelRGB8(prev_idx < 0 ? 0 : 255, inlier ? 0 : 255, 0));
      }
      if (!FLAGS_show_reprojected_pts) {
        for (int i = 0; i < feats.size(); ++i) {
          const KLTFeat& ft = feats[i];
          int prev_idx = FindInSortedArray(prev_ftids, ft.id);
          if (prev_idx >= 0) {
            const Eigen::Vector2f& p = prev_feats[prev_idx].pos;
            const MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(0, 0, 255);
            DrawLine(out, s * p(0), s * p(1), s * ft.pos(0), s * ft.pos(1), c);
            DrawDot(out, s * p(0), s * p(1), c);
          }
//        bool inlier = ftid_pts_map.count(ft.id) > 0;
          bool inlier = binary_search(pose_inliers.begin(), pose_inliers.end(),
                                      ft.id);
          DrawDot(out, s * ft.pos(0), s * ft.pos(1),
                  MakePixelRGB8(prev_idx < 0 ? 0 : 255, inlier ? 0 : 255, 0));
        }
      }
      DrawTextFormat(out, 5, 5, MakePixelRGB8(255, 255, 255), "%03d", idx);
LOG(INFO) << "rgb8 " <<  left.width() << ", " <<  left.height()
    << " - " << out.size();

      vector<char> geom_buf;
      DrawGeometricOutput(vo, &geom_buf);

      csio_out.PushSyncMark(2);
      csio_out.Push(0, out.data(), out.size() * 3);
      csio_out.Push(1, geom_buf.data(), geom_buf.size());
ProfileEnd("3.out", &pdb_);
    }

    prev_ftids = ftids;
    prev_feats = feats;
ProfileEnd("-.frate", &pdb_);
//ProfileDump(pdb_);
  }
  tracker.Cleanup();
  if (ofs_pose_out.is_open()) ofs_pose_out.close();

  return 0;
}

