// csiomod_freenect.cc
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#include "csio_stream.h"
#include "csio_frame_parser.h"

#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <pthread.h>
#include <cmath>

#ifdef APPLE
#include "libfreenect/libfreenect.hpp"
#else
#include <libfreenect.hpp>
#endif

using namespace std;

DEFINE_string(out, "-", "File path for csio::OutputStream (- for stdout).");
DEFINE_int32(kinect_devno, 0, "Kinect device number to use.");
DEFINE_int32(kinect_tilt_angle, 0, "Kinect tilt angle (-30~30 deg.).");

class Mutex {
public:
        Mutex() {
                pthread_mutex_init( &m_mutex, NULL );
        }
        void lock() {
                pthread_mutex_lock( &m_mutex );
        }
        void unlock() {
                pthread_mutex_unlock( &m_mutex );
        }

        class ScopedLock
        {
                Mutex & _mutex;
        public:
                ScopedLock(Mutex & mutex)
                        : _mutex(mutex)
                {
                        _mutex.lock();
                }
                ~ScopedLock()
                {
                        _mutex.unlock();
                }
        };
private:
        pthread_mutex_t m_mutex;
};


class FreenectWrapper : public Freenect::FreenectDevice {
 public:
  FreenectWrapper(freenect_context *_ctx, int _index)
      : Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(640*480), m_buffer_video(freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB).bytes), m_gamma(2048), m_new_rgb_frame(false),
      m_new_depth_frame(false) {
      
      for( unsigned int i = 0 ; i < 2048 ; i++) {
        float v = i/2048.0;
        v = std::pow(v, 3)* 6;
        m_gamma[i] = v*6*256;
      }
  }

  // Do not call directly even in child
  void VideoCallback(void* _rgb, uint32_t timestamp) {
    Mutex::ScopedLock lock(m_rgb_mutex);
    VLOG(1) << "In video callback. " << getVideoBufferSize();
    uint8_t* rgb = static_cast<uint8_t*>(_rgb);
    std::copy(rgb, rgb + getVideoBufferSize(), m_buffer_video.begin());
    m_new_rgb_frame = true;
  }
    
  // Do not call directly even in child
  void DepthCallback(void* _depth, uint32_t timestamp) {
    Mutex::ScopedLock lock(m_depth_mutex);
    VLOG(1) << "In depth callback. " << getDepthBufferSize();
    uint16_t* depth = static_cast<uint16_t*>(_depth);
    std::copy(depth, depth + getDepthBufferSize()/2, m_buffer_depth.begin());
    m_new_depth_frame = true;
  }
    
  bool getRGB(std::vector<uint8_t> &buffer) {
    Mutex::ScopedLock lock(m_rgb_mutex);
    if(!m_new_rgb_frame)
      return false;
    buffer.swap(m_buffer_video);
    m_new_rgb_frame = false;
    return true;
  }
    
  bool getDepth(std::vector<uint16_t> &buffer) {
    Mutex::ScopedLock lock(m_depth_mutex);
    if(!m_new_depth_frame)
      return false;
    buffer.swap(m_buffer_depth);
    m_new_depth_frame = false;
    return true;
  }

  private:
    std::vector<uint16_t> m_buffer_depth;
    std::vector<uint8_t> m_buffer_video;
    std::vector<uint16_t> m_gamma;
    Mutex m_rgb_mutex;
    Mutex m_depth_mutex;
    bool m_new_rgb_frame;
    bool m_new_depth_frame;
};

int main(int argc, char** argv){

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  Freenect::Freenect freenect;
  FreenectWrapper* device = &freenect.createDevice<FreenectWrapper>(0);

  vector<csio::ChannelInfo> channels;
  channels.push_back(csio::ChannelInfo(
          0, csio::MakeImageTypeStr("rgb8", 640, 480), "rgb"));
  channels.push_back(csio::ChannelInfo(
          1, csio::MakeImageTypeStr("gray16", 640, 480), "depth"));
  map<string, string> config;
  LOG(INFO) << "setting up csio::OutputStream (out=" << FLAGS_out << ").";
  csio::OutputStream csio_os;
  if (csio_os.Setup(channels, config, FLAGS_out) == true) {
    LOG(INFO) << "csio::OutputStream opened (out=" << FLAGS_out << "),"
      << "w:640" << "h:480";
  } else {
    return -1;
  }

  device->setDepthFormat(FREENECT_DEPTH_REGISTERED);
  device->startVideo();
  device->startDepth();

  bool die(false);

  std::vector<uint8_t> rgb(640*480*3);
  std::vector<uint16_t> depth(640*480);

  while(!die) {
    if (device->getRGB(rgb) && device->getDepth(depth)) {
      VLOG(1) << "csio send";
      csio_os.PushSyncMark(2);
      csio_os.Push(0, &rgb[0] , 640 * 480 * 3);
      csio_os.Push(1, static_cast<void*>(&depth[0]) , 640 * 480 * 2);
    } else {
    }
  }

  device->stopVideo();
  device->stopDepth();
  return 0;
}
