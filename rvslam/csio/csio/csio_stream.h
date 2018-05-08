// csio_stream.h
//
//  Channeled stream IO based on C++ standard library.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//
// TODO(jwlim): Timestamp does not work.


#ifndef _CSIO_STREAM_H_
#define _CSIO_STREAM_H_

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "csio_channel.h"
#include "csio_time.h"

namespace csio {

class InputStream {
 public:
  InputStream() : isptr_(NULL), isptr_owned_(false) {}
  ~InputStream() {}

  bool Setup(std::istream& is);
  void Close();

  bool Setup(const std::string& filepath) {
    if (filepath.empty()) return false;
    if (filepath == "-") return Setup(std::cin);
    isptr_owned_ = true;
    return Setup(*new std::ifstream(filepath.c_str()));
  }

//  // Doesn't work: rdbuf()->in_avail() == 0 does not mean there is no data
//  // in the buffer - especially with std::cin.
//  bool ReadyToFetch() {
//    return (isptr_ != NULL && isptr_->rdbuf()->in_avail() >= 0);
//  }

  bool Fetch(Frame* frame);
  bool FetchSyncFrames(std::vector<Frame>* frames);

  bool IsOpen() const { return isptr_ != NULL; }

  const std::vector<ChannelInfo>& channels() const { return channels_; }
  const std::map<std::string, std::string>& config() const { return config_; }

 private:
  bool FetchFrameBody(Frame* frame);

  std::istream* isptr_;
  bool isptr_owned_;
  std::vector<ChannelInfo> channels_;
  std::map<std::string, std::string> config_;
};

class OutputStream {
 public:
  OutputStream() : osptr_(NULL), osptr_owned_(false) {}
  ~OutputStream() {}

  bool Setup(const std::vector<ChannelInfo>& channels,
             const std::map<std::string, std::string>& config,
             std::ostream& os);
  void Close();

  bool Setup(const std::vector<ChannelInfo>& channels,
             const std::map<std::string, std::string>& config,
             const std::string& filepath) {
    if (filepath.empty()) return false;
    if (filepath == "-") return Setup(channels, config, std::cout);
    osptr_owned_ = true;
    return Setup(channels, config, *new std::ofstream(filepath.c_str()));
  }

  bool Push(uint32_t chid, const void* buf, uint32_t bufsz,
            double timestamp = -1.0);

  bool Push(const Frame& frame, bool set_current_timestamp = true) {
    return Push(frame.chid, &frame.buf[0], frame.buf.size(),
                set_current_timestamp ? -1.0 : frame.timestamp);
  }

  bool Push(uint32_t chid, std::vector<char>& buf, double timestamp = -1.0) {
    if (timestamp < 0) timestamp = GetTimestamp();
    return Push(chid, &buf[0], buf.size(), timestamp);
  }

  bool PushStr(uint32_t chid, const std::string& str, double timestamp = -1.0) {
    return Push(chid, str.c_str(), str.length() + 1, timestamp);
  }

  bool PushSyncMark(unsigned int num_frames);

  bool PushSyncFrames(const std::vector<Frame>& frames,
                      bool set_current_timestamp = true) {
    bool ret = PushSyncMark(frames.size());
    for (int i = 0; ret && i < frames.size(); ++i) {
      ret = Push(frames[i], set_current_timestamp);
    }
    return ret;
  }

  bool IsOpen() const { return osptr_ != NULL; }

  const std::vector<ChannelInfo>& channels() const { return channels_; }
  const std::map<std::string, std::string>& config() const { return config_; }

 private:
  std::ostream* osptr_;
  bool osptr_owned_;
  std::vector<ChannelInfo> channels_;
  std::map<std::string, std::string> config_;
};

// Utility functions related to character and string operations.

inline std::string EscapeStr(const std::string& str);
inline std::string UnescapeStr(const std::string& str);
inline size_t GetToken(const std::string& str, int pos, std::string* tok,
                       const char* delim = " \t");

inline bool IsNumber(const std::string& str) {
  std::string::const_iterator it = str.begin();
  while (it != str.end() && std::isdigit(*it)) ++it;
  return !str.empty() && it == str.end();
}

// Implementation of the InputStream class.

inline bool InputStream::Setup(std::istream& is) {
  Close();
  std::string str, tok;
  std::getline(is, str);
  if (str.substr(0, 9) != "#CSIO 1.0") return false;
  std::vector<ChannelInfo> channels;
  while (is.good()) {
    std::getline(is, str);
    if (str.empty()) continue;
    if (str.substr(0, 19) == "#CSIO end of header") break;
    if (str[0] == '#') continue;
    size_t pos = GetToken(str, 0, &tok);
    if (IsNumber(tok) == false) return false;
    ChannelInfo chinfo;
    chinfo.id = static_cast<uint32_t>(atoi(tok.c_str()));
    pos = GetToken(str, pos, &tok);
    chinfo.type = UnescapeStr(tok);
    pos = GetToken(str, pos, &tok);
    chinfo.desc = UnescapeStr(tok);
    channels.push_back(chinfo);
  } while (str.substr(0, 19) != "#CSIO end of header");
  isptr_ = &is;
  channels_.swap(channels);
  // TODO: Handle config_ struct.
//  std::map<std::string, std::string> config_;
  return true;
}

inline void InputStream::Close() {
  // If the istream is an ifstream instance, close the file.
  std::ifstream* ifs = dynamic_cast<std::ifstream*>(isptr_);
  if (ifs != NULL) ifs->close();
  if (isptr_owned_) delete isptr_;
  isptr_ = NULL;
  isptr_owned_ = false;
}

inline bool InputStream::Fetch(Frame* frame) {
  if (frame == NULL || isptr_ == NULL || isptr_->good() == false) return false;
  isptr_->read(reinterpret_cast<char*>(&frame->chid), sizeof(uint32_t));
  while (frame->chid >= 0xFFFFFFF0) {  // Special sequence - ignore.
    uint32_t dummy;
    isptr_->read(reinterpret_cast<char*>(&dummy), sizeof(uint32_t));
    isptr_->read(reinterpret_cast<char*>(&frame->chid), sizeof(uint32_t));
  }
  return FetchFrameBody(frame);
}

inline bool InputStream::FetchFrameBody(Frame* frame) {
  uint32_t bufsz;
  isptr_->read(reinterpret_cast<char*>(&frame->timestamp), sizeof(double));
  isptr_->read(reinterpret_cast<char*>(&bufsz), sizeof(uint32_t));
  if (bufsz > 0) {
    frame->buf.resize(bufsz);
    isptr_->read(&frame->buf[0], bufsz);
  }
  return true;
}

inline bool InputStream::FetchSyncFrames(std::vector<Frame>* frames) {
  if (frames == NULL || isptr_ == NULL || isptr_->good() == false) return false;
  const uint32_t SYNC_MARK = 0xFFFFFFF0;
  uint32_t chid, num_frames = 1;
  isptr_->read(reinterpret_cast<char*>(&chid), sizeof(uint32_t));
  if (chid >= 0xFFFFFFF0) {  // Special chids (including SyncMark).
    isptr_->read(reinterpret_cast<char*>(&num_frames), sizeof(uint32_t));
    if (chid != SYNC_MARK) return FetchSyncFrames(frames);
  }
  frames->resize(num_frames);
  if (chid != SYNC_MARK) {
    frames->at(0).chid = chid;
    return FetchFrameBody(&frames->at(0));
  } else {
    bool ret = true;
    for (int i = 0; ret && i < num_frames; ++i) ret = Fetch(&frames->at(i));
    return ret;
  }
}

// Implementation of the OutputStream class.

inline bool OutputStream::Setup(const std::vector<ChannelInfo>& channels,
             const std::map<std::string, std::string>& config,
             std::ostream& os) {
  if (os.good() == false || channels.size() <= 0) return false;
  Close();
  // Output the header to the os.
  os << "#CSIO 1.0" << std::endl;
  for (int i = 0; i < channels.size(); ++i) {
    const ChannelInfo& chinfo = channels[i];
    os << chinfo.id << " " << EscapeStr(chinfo.type) << " "
        << EscapeStr(chinfo.desc) << std::endl;
  }
  // TODO: Output config.
  std::string header_end("#CSIO end of header");
  header_end.resize(256, ' ');  // Leave some space at the end of header.
  os << header_end << std::endl;
  // Setup member variables.
  osptr_ = &os;
  channels_ = channels;
  config_ = config;
  return true;
}

inline void OutputStream::Close() {
  if (osptr_ != NULL) osptr_->flush();
  // If the ostream is a ofstream instance, close the file.
  std::ofstream* ofs = dynamic_cast<std::ofstream*>(osptr_);
  if (ofs != NULL) ofs->close();
  if (osptr_owned_) delete osptr_;
  osptr_ = NULL;
  osptr_owned_ = false;
}

inline bool OutputStream::PushSyncMark(unsigned int num_frames) {
  if (osptr_ == NULL || osptr_->good() == false) return false;
  if (num_frames > 255) return false;
  const uint32_t sync_mark = 0xFFFFFFF0;
  const uint32_t num = num_frames;
  osptr_->write(reinterpret_cast<const char*>(&sync_mark), sizeof(uint32_t));
  osptr_->write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
  osptr_->flush();
  return true;
}

inline bool OutputStream::Push(uint32_t chid, const void* buf, uint32_t bufsz,
                               double timestamp) {
  if (osptr_ == NULL || osptr_->good() == false) return false;
  if (timestamp < 0.0) timestamp = GetTimestamp();
  osptr_->write(reinterpret_cast<const char*>(&chid), sizeof(uint32_t));
  osptr_->write(reinterpret_cast<const char*>(&timestamp), sizeof(double));
  osptr_->write(reinterpret_cast<const char*>(&bufsz), sizeof(uint32_t));
  if (bufsz > 0) osptr_->write(reinterpret_cast<const char*>(buf), bufsz);
  osptr_->flush();
  return true;
}

// Implementation of the utility functions.

inline std::string EscapeStr(const std::string& str) {
  if (str.empty()) return "''";
  if (str.find_first_of(" \t'") == std::string::npos) return str;
  std::string ret = "'";
  size_t i = 0, pos;
  while ((pos = str.find_first_of("'\\", i)) != std::string::npos) {
    ret += str.substr(i, pos - i) + "\\" + str[pos];
    i = pos + 1;
  }
  return ret + str.substr(i, pos - i) + "'";
}

inline std::string UnescapeStr(const std::string& str) {
  if (str.empty() || str[0] != '\'') return str;
  std::string ret = str.substr(1, str.length() - 2);
  size_t i = 0, pos;
  while ((pos = ret.find('\\', i)) != std::string::npos) {
    ret.erase(pos, 1);
    i = pos + 1;
  }
  return ret;
}

inline size_t GetToken(const std::string& str, int pos, std::string* tok,
                       const char* delim /*= " \t"*/) {
  size_t begin = str.find_first_not_of(delim, pos), end = std::string::npos;
  if (begin == std::string::npos) return std::string::npos;
  if (str[begin] == '\'') {
    for (end = begin + 1; str[end] != '\''; ++end) if (str[end] == '\\') ++end;
    ++end;
  } else {
    end = str.find_first_of(delim, begin);
  }
  if (tok) *tok = str.substr(begin, end - begin);
  return end >= str.length() ? std::string::npos : (end + 1);
}

}  // namespace csio
#endif  // _CSIO_STREAM_H_
