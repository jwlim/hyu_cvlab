// vehicle_info.h
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#ifndef _RVSLAM_VEHICLE_INFO_H_
#define _RVSLAM_VEHICLE_INFO_H_

#include <string>
#include <vector>
#include <Eigen/Dense>

namespace rvslam {

class VehicleInfo {
 public:
  // Units: velocigy: km/h -> m/s,  yaw_rate: deg/s -> rad/s,  timestamp: sec.
  struct Item { double velocity, yaw_rate, timestamp; };

  VehicleInfo() : loaded_(false) {}

  bool Load(const std::string& txt_filepath);

  bool loaded() const { return loaded_; }
  size_t size() const { return items_.size(); }
  const Item& operator[](int idx) const { return items_[idx]; }
  const Item* GetItemPtr(int idx) const {
    return (loaded_ && idx >= 0 && idx < size()) ? &items_[idx] : NULL;
  }

  // Compute the nexst 2D pose (x, y, theta) from the current pose and delta.
  static Eigen::Vector3d AddDelta(const Eigen::Vector3d& pose_2d,
                                  double velocity, double yaw_rate,
                                  double time_delta) {
    Eigen::Vector3d ret;
    const double travel = velocity * time_delta;
    const double theta = pose_2d[2] + yaw_rate * time_delta;
    ret[0] = pose_2d[0] + cos(theta) * travel;
    ret[1] = pose_2d[1] + sin(theta) * travel;
    ret[2] = theta;
    return ret;
  }

  static Eigen::Vector3d AddDelta(const Eigen::Vector3d& pose_2d,
                                  const Item& prev, const Item& cur) {
    return AddDelta(pose_2d,
                    cur.velocity, cur.yaw_rate, cur.timestamp - prev.timestamp);
  }

  // The traj contains 2D pose (x, y, theta) on the ground plane.
  void IntegrateEntireTrajectory(
      Eigen::Array<double, 3, Eigen::Dynamic>* traj) const;

 private:
  bool loaded_;
  std::vector<Item> items_;
};

class RelativePose2d {
 public:
  RelativePose2d() : pose_(0.0, 0.0, 0.0), duration_(0), last_timestamp_(0) {}

  bool IsValid() const { return duration_ > 0.0; }
  const Eigen::Vector3d& pose() const { return pose_; }
  double duration() const { return duration_; }
  double last_timestamp() const { return last_timestamp_; }

  void Swap(RelativePose2d* p) {
    pose_.swap(p->pose_);
    std::swap(duration_, p->duration_);
    std::swap(last_timestamp_, p->last_timestamp_);
  }

  void AddVehicleInfoItem(const VehicleInfo::Item& item,
                          double prev_timestamp = -1.0) {
    if (prev_timestamp < 0.0) prev_timestamp = last_timestamp_;
    const double delta = item.timestamp - prev_timestamp;
    pose_ = VehicleInfo::AddDelta(pose_, item.velocity, item.yaw_rate, delta);
    duration_ += delta;
    last_timestamp_ = item.timestamp;
  }

  void ResetPose() {
    pose_ << 0.0, 0.0, 0.0;
    duration_ = 0.0;
  }

 private:
  Eigen::Vector3d pose_;
  double duration_, last_timestamp_;  // Only valid when duration > 0.
};

}  // namespace rvslam
#endif  // _RVSLAM_VEHICLE_INFO_H_
