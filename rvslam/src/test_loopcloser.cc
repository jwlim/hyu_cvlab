#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "time.h"
#include "image_file.h"
#include "image_util.h"
#include "klt_tracker.h"
#include "rvslam_util.h"
#include "visual_odometer.h"
#include "feat_opencv.h"
#include "Keyframe.h"

#include "rvslam_profile.h"
rvslam::ProfileDBType pdb_;

#include "csio/csio_stream.h"
#include "csio/csio_frame_parser.h"

using namespace std;
using namespace rvslam;

DEFINE_string(calib, "",
              "Comma-separated stereo calibration (fx,fy,cx,cy).");
DEFINE_string(image_files, "test.%04d.png", "Input images.");
DEFINE_string(disparity_files, "disparity.%04d.png", "Input disparity.");
DEFINE_string(vehicle_info_path, "vehicle.txt", "Vehicle info.");
DEFINE_string(out, "track_%04d.png", "Tracking result images.");
DEFINE_int32(start, 0, "Start index of files to process.");
DEFINE_int32(end, 1, "End index of files to process.");
DEFINE_int32(step, 1, "Step in index of files to process.");
DEFINE_int32(reduce_size, 0, "Reduce image size.");
DEFINE_int32(max_disparity, 0, "Max disparity value.");
DEFINE_string(feature, "FAST", "feature : FAST, ORB ...");
DEFINE_string(voc_file, "../data/brief_50_3.bin", "vocab tree file");

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
  const vector<Keyframe>& keyframes = vo.keyframes();
  const map<int, Vec3>& ftid_pts_map = vo.ftid_pts_map();
  set<int> kfids;
  if (FLAGS_show_all_keyframes == false) kfposes.clear();
  for (int i = 0; i < keyframes.size(); ++i) {
    const Keyframe& kf = keyframes[i];
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
  

  //Set Calib
  if (FLAGS_calib.empty()) {
    LOG(ERROR) << "calib is not given (fx,fy,cx,cy,k1,k2)";
    return -1;
  }
  VisualOdometer::Calib calib;
  CHECK_EQ(6, sscanf(&FLAGS_calib[0], "%lf,%lf,%lf,%lf,%lf,%lf",
        &calib.fx, &calib.fy, &calib.cx, &calib.cy, &calib.k1, &calib.k2))
      << FLAGS_calib;

  const char* voc_file = FLAGS_voc_file.c_str();
  VisualOdometer vo(calib, voc_file);
  KLTTracker tracker;
  FeatureDetector feat;

  MCImageRGB8 image;
  MCImageGray8 image_gray;

  //Visualization
  csio::OutputStream csio_out;
  ofstream ofs_pose_out;
  if (FLAGS_pose_out.empty() == false) {
    ofs_pose_out.open(FLAGS_pose_out.c_str(), fstream::out);
  }

  vector<int> prev_ftids;
  //vector<KLTFeat> prev_feats;
  Mat3X prev_image_pts;
  map<int, Vec6> kfposes;
  Vec6 prev_pose;
  prev_pose << 0, 0, 0, 0, 0, 0;
  for (int idx = FLAGS_start; idx <= FLAGS_end; idx += FLAGS_step) {
    ProfileBegin("4.Total", &pdb_);
    // Load the images.
    ProfileBegin("0.ImageLoad", &pdb_);
    const string image_path= StringPrintf(FLAGS_image_files.c_str(), idx);
    if (!ReadImageRGB8(image_path, &image)) break;
    RGB8ToGray8(image, &image_gray);
    MCImageGray8::ArrayType image_array = image.GetPlane();
    for (int r = 0; r < FLAGS_reduce_size; ++r) {
      MCImageGray8::ArrayType reduced_array;
      ReduceSize(image_array, &reduced_array);
      image_array.swap(reduced_array);
    }
    //LOG(INFO) << "Frame " << idx << ": "
    //   << image_array.rows() << "x" << image_array.cols();
    ProfileEnd("0.ImageLoad", &pdb_);

    ProfileBegin("1.Feature", &pdb_);
    if (idx == FLAGS_start) {
     feat.Setup(FLAGS_feature, FLAGS_image_files, idx, FLAGS_reduce_size);
    } else {
     int latest_knum = vo.keyframes()[1].frmno;
      feat.Process(FLAGS_feature, FLAGS_image_files, idx, latest_knum,
                   FLAGS_reduce_size, feat.last_ftid());
    }

    //set image_pts
    const vector<int>& ftids = feat.ftids(); 
    const vector<cv::KeyPoint>& keypoints = feat.keypoints();
    Mat3X image_pts(3, ftids.size());
    for (int i = 0; i < ftids.size(); ++i) {
      image_pts.col(i) <<  keypoints[i].pt.x, keypoints[i].pt.y, 1;
    }

    //set descriptors
    cv::Mat descriptors = feat.descriptor(); 

    ProfileEnd("1.Feature", &pdb_);

    ProfileBegin("2.Process", &pdb_);
    vo.ProcessFrame(idx, ftids, image_pts, descriptors);
    ProfileEnd("2.Process", &pdb_);


    //Visualization
    // Setup CSIO output stream if not done.
    ProfileBegin("3.csio", &pdb_);
    if (idx == FLAGS_start && FLAGS_out.empty() == false) {
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
      const Vec6 pose = vo.pose();
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
      const vector<int>& pose_inliers = vo.pose_inliers();

      Mat2X proj_pos;
      //const Mat2X& proj_pos_veh = vo.veh_fpos;
      const Mat2X& proj_pos_p3p = vo.p3p_fpos;
      vo.GetFeatureProjections(ftids, vo.pose(), &proj_pos);
      if (!FLAGS_show_reprojected_pts) {
        for (int i = 0; i < ftids.size(); ++i) {
          const int x = image_pts(0,i);
          const int y = image_pts(1,i);
          int prev_idx = -1;
          for (int k = 0; k < prev_ftids.size(); ++k) {
            int ftid = ftids[i];
            if (prev_ftids[k] == ftid) {
              prev_idx = k;
              break;
            }
          }
          if (prev_idx >= 0) {
            const Vec3& p = prev_image_pts.col(prev_idx);
            const MCImageRGB8::PixelType c = MCImageRGB8::MakePixel(0, 0, 255);
            DrawLine(out, s * p(0), s * p(1), s * x, s * y, c);
            DrawDot(out, s * p(0), s * p(1), c);
          }

          bool inlier = false;
          for (int k = 0; k < pose_inliers.size(); ++k) {
            int ftid = pose_inliers[k];
            if (ftid == ftids[i]) {
              inlier = true;
            }
          }
          DrawDot(out, s * x, s * y,
                  MakePixelRGB8(prev_idx < 0 ? 0 : 255, inlier ? 0 : 255, 0));
        }
      }
      DrawTextFormat(out, 5, 5, MakePixelRGB8(255, 255, 255), "%03d", idx);
LOG(INFO) << "rgb8 " <<  image.width() << ", " <<  image.height()
    << " - " << out.size();

      vector<char> geom_buf;
      DrawGeometricOutput(vo, &geom_buf);

      csio_out.PushSyncMark(2);
      csio_out.Push(0, out.data(), out.size() * 3);
      csio_out.Push(1, geom_buf.data(), geom_buf.size());
    }

    prev_image_pts = image_pts;
    prev_ftids = ftids;
    ProfileEnd("3.csio", &pdb_);
    ProfileEnd("4.Total", &pdb_);

   //ProfileDump(pdb_);
  }


  return 0;
}

