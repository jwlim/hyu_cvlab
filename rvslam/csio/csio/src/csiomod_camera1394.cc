#include "csiomod_camera1394.h"

#include <iostream>
#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
using namespace std;

dc1394camera_t* GetDevice(dc1394_t *d, int32_t devno) {
  dc1394error_t err;
  dc1394camera_list_t *list;

  if (dc1394_camera_enumerate(d, &list) != DC1394_SUCCESS) {
    LOG(ERROR) << "Failed to enumerate cameras";
    exit(1);
  }

  if (list->num == 0) {
    LOG(ERROR) << "No cameras found";
    exit(1);
  }

  if (devno > list->num - 1) {
    LOG(ERROR) << "Device number is out of index.";
    exit(1);
  }
  
  dc1394camera_t *camera = dc1394_camera_new(d, list->ids[devno].guid);
  if (!camera) {
    LOG(ERROR) << "Failed to initialize camera with guid %llx",
      list->ids[devno].guid;
    exit(1);
  }
  dc1394_camera_free_list (list);

  return camera;
}

void ListDev(dc1394_t *d) {
  dc1394error_t err;
  dc1394camera_list_t *list;

  if (dc1394_camera_enumerate(d, &list) != DC1394_SUCCESS) {
    LOG(ERROR) << "Failed to enumerate cameras";
    exit(1);
  }

  if (list->num == 0) {
    LOG(ERROR) << "No cameras found";
    exit(1);
  }

  vector<dc1394camera_t*> camera_list(list->num);
  printf("# Camera list [%d]\n", camera_list.size());
  for (uint32_t i = 0; i<list->num; i++) {
    dc1394camera_t *camera = dc1394_camera_new(d, list->ids[i].guid);
    if (!camera) {
      LOG(ERROR) << "Failed to initialize camera with guid %llx",
        list->ids[i].guid;
      exit(1);
    }
    camera_list.push_back(camera);
    printf("%d. vendor: '%s'\n   model: '%s'\n   id: '%08lx.%08lx'\n",
      i, camera->vendor, camera->model, (uint32_t)(camera->guid>>32),
      camera->guid);
  }

  int input;
  printf("\nInput cam number : ");
  scanf("%d", &input);
  if (input >= list->num || input < 0) {
    printf("Error : out of index.\n");
  }

  dc1394camera_t *selected_cam = *(camera_list.begin()+1+input);
  vector<dc1394camera_t*>::iterator it;
  for (it = camera_list.begin()+1; it!=camera_list.end(); it++) {
    if (*it != selected_cam) {
      dc1394_camera_free(*it);
    }
  }
  dc1394_camera_free_list (list);
  
  Camera1394 cam1394(selected_cam);
  printf("\n# Features of '%s'\n", cam1394.cam->model);
  cam1394.ListAvailableFeatures();

  printf("\n# Supported modes\n");
  cam1394.ListSupportedModes();

  dc1394_camera_free(cam1394.cam);
  dc1394_free (d);
}

Camera1394::Camera1394(dc1394camera_t *cam)
: cam(cam) {}

Camera1394::~Camera1394(){}

void Camera1394::ListAvailableFeatures()
{
  dc1394featureset_t features;
  dc1394_feature_get_all(cam, &features);
  for (uint32_t i=0; i<DC1394_FEATURE_NUM; i++) {
    if (features.feature[i].available) {
      dc1394feature_info_t *feature = &features.feature[i];
      if (feature->id == DC1394_FEATURE_TRIGGER) {
        printf("   %s : src.mode[.polarity] #", g_featurenames[i]);
        for (uint32_t k=0; i<feature->trigger_sources.num; k++)
          printf("%s%d", k?",":" src=",
            feature->trigger_sources.sources[k] - DC1394_TRIGGER_SOURCE_0);
        for (uint32_t k=0; k<feature->trigger_modes.num; k++)
          printf("%s%d", k?",":" src=",
            feature->trigger_modes.modes[k] - DC1394_TRIGGER_MODE_0);
        printf(" polarity = %d\n", (int)feature->trigger_polarity);
      } else {
        dc1394feature_modes_t modes;
        dc1394_feature_get_modes(cam, feature->id, &modes);
        printf("   %s:", g_featurenames[i]);
        std::string str;
        for (uint32_t k=0; k<modes.num; k++)
          switch (modes.modes[k]) {
            case DC1394_FEATURE_MODE_AUTO:  str=" auto"; break;
            case DC1394_FEATURE_MODE_ONE_PUSH_AUTO:  str=" onepush"; break;
            case DC1394_FEATURE_MODE_MANUAL:  str =""; break;
        }
        printf("%s", str.c_str());
        if (feature->id == DC1394_FEATURE_WHITE_BALANCE) {
          uint32_t u, v;
          dc1394_feature_whitebalance_get_value(cam, &u, &v);
          printf(" %d.%d(%d~%d)", u,v, feature->min, feature->max);
        } else
          printf(" %d(%d~%d)",
            feature->value, feature->min, feature->max);
        if (feature->absolute_capable)
          printf(" = @%.3f(%.3f~%.3f)\n", feature->abs_value,
            feature->abs_min, feature->abs_max);
        else
          printf("\n");
      }
    }
  }
}

void Camera1394::ListSupportedModes()
{
  dc1394video_modes_t modes;
  dc1394framerates_t frates;
  std::string str;
  int input;

  dc1394_video_get_supported_modes(cam, &modes);
  for (uint32_t i=0; i<modes.num; i++) {
    dc1394video_mode_t mode = modes.modes[i];
    switch (mode) {
      case DC1394_VIDEO_MODE_160x120_YUV444:  str="160x120.yuv444";  break;
      case DC1394_VIDEO_MODE_320x240_YUV422:  str="320x240.yuv422";  break;
      case DC1394_VIDEO_MODE_640x480_YUV411:  str="640x480.yuv411";  break;
      case DC1394_VIDEO_MODE_640x480_YUV422:  str="640x480.yuv422";  break;
      case DC1394_VIDEO_MODE_640x480_RGB8:  str="640x480.rgb8";  break;
      case DC1394_VIDEO_MODE_640x480_MONO8:  str="640x480.mono8";  break;
      case DC1394_VIDEO_MODE_640x480_MONO16:  str="640x480.mono16";  break;
      case DC1394_VIDEO_MODE_800x600_YUV422:  str="800x600.yuv422";  break;
      case DC1394_VIDEO_MODE_800x600_RGB8:  str="800x600.rgb8";  break;
      case DC1394_VIDEO_MODE_800x600_MONO8:  str="800x600.mono8";  break;
      case DC1394_VIDEO_MODE_800x600_MONO16:  str="800x600.mono16";  break;
      case DC1394_VIDEO_MODE_1024x768_YUV422:  str="1024x768.yuv422";  break;
      case DC1394_VIDEO_MODE_1024x768_RGB8:  str="1024x768.rgb8";  break;
      case DC1394_VIDEO_MODE_1024x768_MONO8:  str="1024x768.mono8";  break;
      case DC1394_VIDEO_MODE_1024x768_MONO16:  str="1024x768.mono16";  break;
      case DC1394_VIDEO_MODE_1280x960_YUV422:  str="1280x960.yuv422";  break;
      case DC1394_VIDEO_MODE_1280x960_RGB8:  str="1280x960.rgb8";  break;
      case DC1394_VIDEO_MODE_1280x960_MONO8:  str="1280x960.mono8";  break;
      case DC1394_VIDEO_MODE_1280x960_MONO16:  str="1280x960.mono16";  break;
      case DC1394_VIDEO_MODE_1600x1200_YUV422:  str="1600x1200.yuv422";  break;
      case DC1394_VIDEO_MODE_1600x1200_RGB8:  str="1600x1200.rgb8";  break;
      case DC1394_VIDEO_MODE_1600x1200_MONO8:  str="1600x1200.mono8";  break;
      case DC1394_VIDEO_MODE_1600x1200_MONO16:  str="1600x1200.mono16";  break;
      case DC1394_VIDEO_MODE_EXIF:  str="exif";  break;
      case DC1394_VIDEO_MODE_FORMAT7_0:  str="format7.0";  break;
      case DC1394_VIDEO_MODE_FORMAT7_1:  str="format7.1";  break;
      case DC1394_VIDEO_MODE_FORMAT7_2:  str="format7.2";  break;
      case DC1394_VIDEO_MODE_FORMAT7_3:  str="format7.3";  break;
      case DC1394_VIDEO_MODE_FORMAT7_4:  str="format7.4";  break;
      case DC1394_VIDEO_MODE_FORMAT7_5:  str="format7.5";  break;
      case DC1394_VIDEO_MODE_FORMAT7_6:  str="format7.6";  break;
      case DC1394_VIDEO_MODE_FORMAT7_7:  str="format7.7";  break;
      default: str="unknown mode";
    }

    printf("%d. mode: %s [%d]\n", i, str.c_str(), (unsigned int) mode);

    if (mode >= DC1394_VIDEO_MODE_FORMAT7_MIN && 
        mode <= DC1394_VIDEO_MODE_FORMAT7_MAX) {
      dc1394format7mode_t f7;
      dc1394_format7_get_mode_info(cam, mode, &f7);
      printf("   colorcoding : %s (%s", 
        g_colorcodingnames[f7.color_coding - DC1394_COLOR_CODING_MIN],
        g_colorcodingnames[f7.color_codings.codings[0] -
          DC1394_COLOR_CODING_MIN]);
      for (unsigned int k=1; k<f7.color_codings.num; k++){
        printf(", %s", g_colorcodingnames[f7.color_codings.codings[k] -
          DC1394_COLOR_CODING_MIN]);
      }
      printf(")\n   roi: %d,%d/%d,%d %dx%d(~%dx%d/%dx%d) bpp: %d(~%d/%d)\n",
        f7.pos_x,f7.pos_y, f7.unit_size_x,f7.unit_size_y,
        f7.size_x,f7.size_y, f7.max_size_x,f7.max_size_y,
        f7.unit_pos_x,f7.unit_pos_y,
        f7.packet_size, f7.max_packet_size, f7.unit_packet_size);
    } else {
      dc1394_video_get_supported_framerates(cam, mode, &frates);
      for (uint32_t j=0; j<frates.num; j++) {
        switch (frates.framerates[j]) {
          case DC1394_FRAMERATE_1_875:  str = "1.875";  break;
          case DC1394_FRAMERATE_3_75:  str = "3.75";  break;
          case DC1394_FRAMERATE_7_5:  str = "7.5";  break;
          case DC1394_FRAMERATE_15:  str = "15";  break;
          case DC1394_FRAMERATE_30:  str = "30";  break;
          case DC1394_FRAMERATE_60:  str = "60";  break;
          case DC1394_FRAMERATE_120:  str = "120";  break;
          case DC1394_FRAMERATE_240:  str = "240";  break;
        }
        printf("   framerate : %s [%d]\n", str.c_str(), 
          (unsigned int)(frates.framerates[j] - DC1394_FRAMERATE_1_875));
      }
    }
  }
}

bool Camera1394::ParseMode(string modestr, string fratestr)
{
  if (modestr=="160x120.yuv444") mode=DC1394_VIDEO_MODE_160x120_YUV444;
  else if (modestr=="320x240.yuv422") mode=DC1394_VIDEO_MODE_320x240_YUV422;
  else if (modestr=="640x480.yuv411") mode=DC1394_VIDEO_MODE_640x480_YUV411;
  else if (modestr=="640x480.yuv422") mode=DC1394_VIDEO_MODE_640x480_YUV422;
  else if (modestr=="640x480.rgb8") mode=DC1394_VIDEO_MODE_640x480_RGB8;
  else if (modestr=="640x480.mono8") mode=DC1394_VIDEO_MODE_640x480_MONO8;
  else if (modestr=="640x480.mono16") mode=DC1394_VIDEO_MODE_640x480_MONO16;
  else if (modestr=="800x600.yuv422") mode=DC1394_VIDEO_MODE_800x600_YUV422;
  else if (modestr=="800x600.rgb8") mode=DC1394_VIDEO_MODE_800x600_RGB8;
  else if (modestr=="800x600.mono8") mode=DC1394_VIDEO_MODE_800x600_MONO8;
  else if (modestr=="800x600.mono16") mode=DC1394_VIDEO_MODE_800x600_MONO16;
  else if (modestr=="1024x768.yuv422") mode=DC1394_VIDEO_MODE_1024x768_YUV422;
  else if (modestr=="1024x768.rgb8") mode=DC1394_VIDEO_MODE_1024x768_RGB8;
  else if (modestr=="1024x768.mono8") mode=DC1394_VIDEO_MODE_1024x768_MONO8;
  else if (modestr=="1024x768.mono16") mode=DC1394_VIDEO_MODE_1024x768_MONO16;
  else if (modestr=="1280x960.yuv422") mode=DC1394_VIDEO_MODE_1280x960_YUV422;
  else if (modestr=="1280x960.rgb8") mode=DC1394_VIDEO_MODE_1280x960_RGB8;
  else if (modestr=="1280x960.mono8") mode=DC1394_VIDEO_MODE_1280x960_MONO8;
  else if (modestr=="1280x960.mono16") mode=DC1394_VIDEO_MODE_1280x960_MONO16;
  else if (modestr=="1600x1200.yuv422") 
    mode=DC1394_VIDEO_MODE_1600x1200_YUV422;
  else if (modestr=="1600x1200.rgb8") mode=DC1394_VIDEO_MODE_1600x1200_RGB8;
  else if (modestr=="1600x1200.mono8") mode=DC1394_VIDEO_MODE_1600x1200_MONO8;
  else if (modestr=="1600x1200.mono16") 
    mode=DC1394_VIDEO_MODE_1600x1200_MONO16;
  else if (modestr=="exif") mode=DC1394_VIDEO_MODE_EXIF;
  else
    return false;

  if (fratestr=="1.875")  frate=DC1394_FRAMERATE_1_875;
  else if (fratestr=="3.75")  frate=DC1394_FRAMERATE_3_75;
  else if (fratestr=="7.5")  frate=DC1394_FRAMERATE_7_5;
  else if (fratestr=="15")  frate=DC1394_FRAMERATE_15;
  else if (fratestr=="30")  frate=DC1394_FRAMERATE_30;
  else if (fratestr=="60")  frate=DC1394_FRAMERATE_60;
  else if (fratestr=="120")  frate=DC1394_FRAMERATE_120;
  else if (fratestr=="240")  frate=DC1394_FRAMERATE_240;
  else if (fratestr!="modeset")
    return false;
  return true;
}

bool Camera1394::ParseFmt7Mode(string modestr)
{
  string str = modestr, tok = strtok(const_cast<char *>(str.c_str()), ".");

  if (tok != "fmt7" || str.empty())
    return false;
  // "\n# example: fmt7.0.raw8.100.640.480.4.4\n"
  mode = (dc1394video_mode_t)(DC1394_VIDEO_MODE_FORMAT7_0 +
    _toi(tok=strtok(NULL,".")));

  if (dc1394_format7_get_mode_info(cam, mode, &f7) != DC1394_SUCCESS) {
    return false;
  }

  tok = strtok(NULL, ".");
  if (tok == "mono8") color_coding=DC1394_COLOR_CODING_MONO8,
    codingstr="gray8";
  else if (tok=="yuv411") color_coding=DC1394_COLOR_CODING_YUV411,
    codingstr="yuv411";
  else if (tok=="yuv422") color_coding=DC1394_COLOR_CODING_YUV422,
    codingstr="yuv422";
  else if (tok=="yuv444")  color_coding=DC1394_COLOR_CODING_YUV444,
    codingstr="yuv444";
  else if (tok=="rgb8")   color_coding=DC1394_COLOR_CODING_RGB8,
    codingstr="rgb8";
  else if (tok=="mono16")  color_coding=DC1394_COLOR_CODING_MONO16,
   codingstr="gray16";
  else if (tok=="rgb16")   color_coding=DC1394_COLOR_CODING_RGB16,
    codingstr="rgb16";
  else if (tok=="mono16s") color_coding=DC1394_COLOR_CODING_MONO16S,
   codingstr="gray16s";
  else if (tok=="rgb16s")  color_coding=DC1394_COLOR_CODING_RGB16S,
   codingstr="rgb16s";
  else if (tok=="raw8")    color_coding=DC1394_COLOR_CODING_RAW8,
     codingstr="raw8";
  else if (tok=="raw16")   color_coding=DC1394_COLOR_CODING_RAW16,
    codingstr="raw16";
  else  return false;

  if (!str.empty())  w = _toi(tok=strtok(NULL, "."));
  if (!str.empty())  h = _toi(tok=strtok(NULL, "."));
  if (!str.empty())  x = _toi(tok=strtok(NULL, "."));
  if (!str.empty())  y = _toi(tok=strtok(NULL, "."));
//  if (!str.empty())  speed = _tof(str);
  return true;
}

bool Camera1394::Setup(string modestr, string fratestr)
{
  bool success = false;
  if (strncmp(modestr.c_str(), "fmt7", 4) != 0) {
    success = ParseMode(modestr, fratestr);
  } else {
    success = ParseFmt7Mode(modestr);
  }
  
  if (!success) {
    LOG(ERROR) << "Failed to parse mode";
    return false;
  }

  dc1394_video_set_iso_speed(cam, DC1394_ISO_SPEED_400);
  if (IsFmt7()) {
    if (dc1394_video_set_mode(cam, mode) != DC1394_SUCCESS) {
      LOG(ERROR) << "Failed to set mode";
      return false;
    }

    if (w <=0 || h <=0) {
      if (dc1394_format7_set_color_coding(
        cam, mode, color_coding) != DC1394_SUCCESS) {
        LOG(ERROR) << "Failed to set color coding";
        return false;
      }
    } else {
      if (dc1394_format7_set_roi(cam, mode, color_coding, DC1394_USE_MAX_AVAIL,
        x, y, w, h) != DC1394_SUCCESS) {
        LOG(ERROR) << "Failed to set roi";
        return false;
      }
    }
  } else {
    if (dc1394_video_set_mode(cam, mode) != DC1394_SUCCESS) {
      LOG(ERROR) << "Failed to set mode";
      return false;
    }
    if (dc1394_video_set_framerate(cam, frate) != DC1394_SUCCESS) {
      LOG(ERROR) << "Failed to set framerate";
      return false;
    }
  }

  if (dc1394_capture_setup(cam, 4, DC1394_CAPTURE_FLAGS_DEFAULT)
      != DC1394_SUCCESS) {
    LOG(ERROR) << "Failed to setup capture";
    return false;
  }
  return true;
}

bool Camera1394::StartCapture()
{
  if (dc1394_video_set_transmission(cam, DC1394_ON) != DC1394_SUCCESS) {
    LOG(ERROR) << "Unable to start camera iso transmission";
    dc1394_capture_stop(cam);
    dc1394_camera_free(cam);
    return false;
  }
  return true;
}

bool Camera1394::StopCapture()
{
  if (dc1394_video_set_transmission(cam, DC1394_OFF) != DC1394_SUCCESS) {
    LOG(ERROR) << "Couldn't stop the camera";
    dc1394_capture_stop(cam);
    dc1394_camera_free(cam);
    return false;
  }
  return true;
}

bool Camera1394::FetchFrame(dc1394video_frame_t **frame)
{
  if (dc1394_capture_dequeue(cam, DC1394_CAPTURE_POLICY_WAIT,
      frame) != DC1394_SUCCESS) {
    LOG(ERROR) << "Unable to capture";
    dc1394_capture_stop(cam);
    dc1394_camera_free(cam);
    return false;
  }
  return true;
}

bool Camera1394::DropFrame(dc1394video_frame_t **frame)
{
  if (dc1394_capture_enqueue(cam, *frame) != DC1394_SUCCESS) {
    LOG(ERROR) << "Unable to drop frame";
    dc1394_capture_stop(cam);
    dc1394_camera_free(cam);
    return false;
  }
  return true;
}

bool Camera1394::IsFmt7()
{
  if (mode >= DC1394_VIDEO_MODE_FORMAT7_MIN && 
      mode <= DC1394_VIDEO_MODE_FORMAT7_MAX) {
    return true;
  } else {
    return false;
  }
}
