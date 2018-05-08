#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "rvslam_common.h"
#include "rvslam_util.h"

using namespace std;
using namespace rvslam;

namespace {

TEST(RVSLAMUtilTest, TestHomEuc) {
  Vec2 v2 = (Vec2() << 0, 0).finished();
  EXPECT_EQ((Vec3() << 0, 0, 1).finished(), Hom(v2));
  EXPECT_EQ(v2, Euc(Hom(v2)));

  Mat2X m2(2, 2);
  m2 << 0, 1, 2, 3;
  EXPECT_EQ((Mat3X(3, 2) << 0, 1, 2, 3, 1, 1).finished(), Hom(m2));

  Vec3 v3 = (Vec3() << 1, 2, 2).finished();
  EXPECT_EQ((Vec2() << 0.5, 1).finished(), Euc(v3));

  Mat3X m3(3, 2);
  m3 << 1, 2, 3, 4, 2, 4;
  EXPECT_EQ((Mat2X(2, 2) << 0.5, 0.5, 1.5, 1.0).finished(), Euc(m3));
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}

