// image_util.cc
//

#include "image_util.h"

using namespace std;

namespace rvslam {

uint8_t _g_fontimgdata[] = {
  0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,
  0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,
  0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,
  0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,1,1,0,
  0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,
  0,1,1,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,0,1,1,1,0,
  0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,
  1,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,
  0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,

  0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,1,0,1,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,
  0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,
  0,1,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,
  0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,
  1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,
  0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,

  0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,1,0,0,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,
  0,0,0,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,
  0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,
  0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,1,0,
  0,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,
  0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

  0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,
  0,0,0,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,
  0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,
  0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,
  0,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,1,0,0,1,1,1,0,0,1,0,1,1,0,1,1,1,1,
  0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,1,0,
  0,1,0,0,1,0,1,1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0,1,1,0,
  0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,
  0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,1,1,1,1,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,
  0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,
  0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

  0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,
  0,0,0,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,
  0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,
  0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,1,
  0,1,1,1,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,
  0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,0,0,
  0,1,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,
  0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,1,1,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,1,1,1,0,0,1,0,0,0,1,0,0,1,0,1,0,1,0,
  0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,
  0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

  0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,
  0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,
  0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,1,1,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,
  0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,
  0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,
  0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,
  0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,1,1,1,1,0,1,0,0,1,
  0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,1,0,
  0,0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,
  0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,1,1,1,0,0,0,1,1,0,
  0,0,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

  0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,
  0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,
  0,1,1,1,0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,
  0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,
  0,1,0,0,1,0,1,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,
  0,1,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,1,1,1,0,1,0,0,1,0,1,0,0,1,0,1,1,1,1,0,1,0,0,0,
  0,0,1,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,
  0,0,1,0,0,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,
  0,0,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,1,0,
  0,0,1,0,0,0,0,0,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,1,1,0,1,0,0,1,0,0,1,1,0,0,1,1,1,0,
  0,0,1,1,1,0,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,0,1,0,0,1,0,0,1,
  0,0,0,0,1,0,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,
  0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
  0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

}
