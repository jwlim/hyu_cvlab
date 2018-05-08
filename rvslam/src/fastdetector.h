// fastdetector.h

#ifndef _RVSLAM_FAST_DETECTOR_H_
#define _RVSLAM_FAST_DETECTOR_H_


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <glog/logging.h>

#include "rvslam_common.h"


namespace rvslam {

void DetectFastFeature(const unsigned char* im, int xsize, int ysize, int stride, int fast_th,
                       int block, std::vector<cv::KeyPoint>* result);

} // namespace

#endif // _RVSLAM_FAST_DETECTOR_H_
