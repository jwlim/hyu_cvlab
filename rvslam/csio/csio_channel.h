// csio_channel.h
//
//  Channeled stream IO based on C++ standard library.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _CSIO_CHANNEL_H_
#define _CSIO_CHANNEL_H_

#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace csio {

struct ChannelInfo {
  uint32_t id;
  std::string type;  // Similar to MIME type string: 'image/x-raw;width=...'
  std::string desc;

  ChannelInfo() : id(), type(), desc() {}
  ChannelInfo(uint32_t id, const std::string& type, const std::string& desc)
      : id(id), type(type), desc(desc) {}
};

struct Frame {
  uint32_t chid;  // Channel id.
  double timestamp;  // Seconds after the stream beginning.
  std::vector<char> buf;

  // chid of 0xFFFFFFF0-0xFFFFFFFF is reserved for special functions.
  // The frame with these ids will not be transmitted.
  Frame() : chid(0xFFFFFFFF), timestamp(-1.0), buf() {}
  Frame(int chid, double ts = -1.0) : chid(chid), timestamp(ts), buf() {}
  Frame(const Frame& frm)
      : chid(frm.chid), timestamp(frm.timestamp), buf(frm.buf) {}
  void Copy(const Frame& frm) {
    chid = frm.chid, timestamp = frm.timestamp, buf = frm.buf;
  }
};

inline std::ostream& operator<<(std::ostream& os, const ChannelInfo& chinfo) {
  os << chinfo.id << ": " << chinfo.type << " (" << chinfo.desc << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Frame& frame) {
  const int sec = frame.timestamp;
  const int ms = (frame.timestamp - sec) * 1000;
  os << "frame " << frame.chid << ": " << frame.buf.size() << " bytes @ "
     << sec << "." << std::setw(3) << std::setfill('0') << ms;
  return os;
}

}  // namespace csio
#endif  // _CSIO_CHANNEL_H_
