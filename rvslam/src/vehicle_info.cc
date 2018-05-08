// vehicle_info.cc
//
// Author: Jongwoo Lim (jongwoo.lim@gmail.com)

#include "vehicle_info.h"
#include <cmath>
#include <fstream>

using namespace std;

namespace rvslam {

bool VehicleInfo::Load(const std::string& txt_filepath) {
  ifstream ifs(txt_filepath.c_str());
  loaded_ = false;
  if (ifs.is_open() == false) return false;
  items_.clear();
  const double kph_to_mps = 1000.0 / (60 * 60);
  const double deg_to_rad = M_PI / 180.0;
  // For debugging : assume 30 fps.
  const double ts_delta = 1.0 / 30;  // Debug.
  double timestamp = 0.0;  // Debug.
  while (ifs.good()) {
    Item item;
    ifs >> item.velocity >> item.yaw_rate >> item.timestamp;
    item.velocity *= kph_to_mps;
    item.yaw_rate *= deg_to_rad;
    item.timestamp = timestamp;  // Debug.
    timestamp += ts_delta;  // Debug.
    items_.push_back(item);
  }
  ifs.close();
  return loaded_ = true;
}

void VehicleInfo::IntegrateEntireTrajectory(
    Eigen::Array<double, 3, Eigen::Dynamic>* traj_ptr) const {
  Eigen::Array<double, 3, Eigen::Dynamic>& traj = *traj_ptr;
  traj.resize(3, items_.size());
  traj.col(0).fill(0.0);
  // TODO: Clarify whether (i-1, i) or (i, i+1).
  double theta = 0.0;
  double timestamp = items_[0].timestamp;
  for (int i = 1; i < items_.size(); ++i) {
    const Item& cur = items_[i];
    const double delta = cur.timestamp - items_[i - 1].timestamp;
    const double travel = cur.velocity * delta;
    theta += cur.yaw_rate * delta;
    traj(0, i) = traj(0, i - 1) + cos(theta) * travel;
    traj(1, i) = traj(1, i - 1) + sin(theta) * travel;
    traj(2, i) = theta;
  }
}

}  // namespace rvslam
