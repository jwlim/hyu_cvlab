#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <iostream>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <Eigen/Dense>

#include "csio_channel.h"
#include "csio_stream.h"
#include "csio_util.h"
#include "csio_frame_parser.h"

using namespace std;

DEFINE_string(in, "-", "File path for csio::InputStream (- for stdin).");
DEFINE_int32(num_capture,1,"the number of capture ");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

//initialize/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////  
  LOG(INFO) << "setting up csio::Inputstream (in=" << FLAGS_in << ").";
  csio::InputStream csin;
  if (csin.Setup(FLAGS_in) == false) {
    LOG(ERROR) << "failed to open csio::InputStream (in=" << FLAGS_in << ").";
    return -1;
  }
  LOG(INFO) << "setup csio::InputStream(" << FLAGS_in << ") complete.";

  const vector<csio::ChannelInfo>& channels = csin.channels();

  if (channels.size() !=2 ) {
    LOG(ERROR) << "channel's size should be 2 ( cam,imu)";
    return -1 ;
  } 

  const csio::ChannelInfo& ch_info_cam = channels[0];
  const csio::ChannelInfo& ch_info_imu = channels[1];

  map <string, string> cfg_cam;
  string type_cam = csio::ParseTypeStr(ch_info_cam.type, &cfg_cam);
  string pixel_type;
  int w = 640;
  int h = 480;
   
  map <string, string> cfg_imu;
  string type_imu = csio::ParseTypeStr(ch_info_imu.type, &cfg_imu);
  
  if (!csio::IsImageType(type_cam, cfg_cam,&pixel_type,&w,&h)) {
    LOG(ERROR) << "unknown cam_type '"<<type_cam <<"'.";
    return -1;
  }
  if (!csio::IsIMUType(type_imu, cfg_imu)) {
    LOG(ERROR) << "unknown imu_type '" << type_imu << "'.";
    return -1;
  }
///////////////////////////////////////////////////////////////////////////////  


  vector<csio::Frame> frame_array_ptr;

  for (int k = 0; k<FLAGS_num_capture; k++) {
    
    //get a frame here one by one
    if (csin.FetchSyncFrames(&frame_array_ptr) == false) {
      break;    
    }
    csio::Frame frame_cam = frame_array_ptr.at(0);
    uint8_t* data_cam = reinterpret_cast<uint8_t*>(frame_cam.buf.data());
   
    //image matrix  
    Eigen::MatrixXd img_mat(h,w); 

    for(int row = 0; row < h; row++) 
      for(int col = 0 ; col < w; col++) 
        img_mat(row,col) = data_cam[row*w+col];
    
    //imu data   
    csio::Frame frame_imu = frame_array_ptr.at(1);
    char* data_imu = (frame_imu.buf.data()); 

    cout <<data_imu<<endl;
    string token = strtok(data_imu,",");  
    
    double roll = atof(token.c_str());
    double pitch = atof((token=strtok(NULL, ",")).c_str());
    double yaw = atof((token=strtok(NULL, ",")).c_str());
    double accX = atof((token=strtok(NULL, ",")).c_str());
    double accY = atof((token=strtok(NULL, ",")).c_str());
    double accZ = atof((token=strtok(NULL, ",")).c_str());
     
  }
}

