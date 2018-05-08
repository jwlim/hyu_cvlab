// rvslam_time.h
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _RVSLAM_TIME_H_
#define _RVSLAM_TIME_H_

#if defined(WIN32) && !defined(CYGWIN)

#include <windows.h>
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_100NANOSECS 116444736000000000Ui64
#else
#define DELTA_EPOCH_IN_100NANOSECS 116444736000000000ULL
#endif

namespace rvslam {

inline double GetTimestamp() {
  // From MSDN answer to gettimeofday.
  FILETIME ft;
  ::GetSystemTimeAsFileTime(&ft);
  unsigned __int64 t = ((ft.dwHighDateTime << 32) | ft.dwLowDateTime)
      - DELTA_EPOCH_IN_100NANOSECS;  // converting file time to unix epoch
  return static_cast<double>(t) / 10000000;
}

inline void GetTimestamp(unsigned int* sec, unsigned int* usec) {
  // From MSDN answer to gettimeofday.
  FILETIME ft;
  ::GetSystemTimeAsFileTime(&ft);
  unsigned __int64 t = ((ft.dwHighDateTime << 32) | ft.dwLowDateTime)
      - DELTA_EPOCH_IN_100NANOSECS;  // converting file time to unix epoch
  *sec = t / 10000000;
  *usec = (t / 10) % 1000000;
}

}  // namespace rvslam

#else  // defined(WIN32) && !defined(CYGWIN)

#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>

namespace rvslam {

inline double GetTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

inline void GetTimestamp(unsigned int* sec, unsigned int* usec) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  *sec = tv.tv_sec;
  *usec = tv.tv_usec;
}

}  // namespace rvslam

#endif  // defined(WIN32) && !defined(CYGWIN)

#include <iostream>
#include <iomanip>
#include <map>
#include <string>

namespace rvslam {

struct ProfileInfo {
  int count;
  double total, min, max, ts, diff;
  ProfileInfo() : count(0), total(0.0), min(1e9), max(0.0), ts(0.0) {}
};
typedef std::map<std::string, ProfileInfo> ProfileDBType;

inline void ProfileBegin(const std::string& tag, ProfileDBType* pdb) {
  (*pdb)[tag].ts = GetTimestamp();
}

inline void ProfileEnd(const std::string& tag, ProfileDBType* pdb) {
  const double ts = GetTimestamp();
  ProfileInfo& info = (*pdb)[tag];
  const double diff = ts - info.ts;
  info.count += 1;
  info.total += diff;
  info.diff = diff;
  if (info.max < diff) info.max = diff;
  if (info.min > diff) info.min = diff;
}

inline void ProfileDump(const ProfileDBType& pdb) {
  int max_len = 0;
  for (ProfileDBType::const_iterator it = pdb.begin(); it != pdb.end(); ++it) {
    if (max_len < it->first.size()) max_len = it->first.size();
  }
  for (ProfileDBType::const_iterator it = pdb.begin(); it != pdb.end(); ++it) {
    const ProfileInfo& info = it->second;
    std::cerr << std::setw(max_len) << std::left << it->first << ":\t"
        << std::fixed << std::setprecision(3) << std::setw(6) << std::right
        << "time : " << info.diff * 1000 << ", mean : " << info.total / info.count * 1000 << " (" << info.count << ") "
        << info.min * 1000 << " ~ " << info.max * 1000 << std::endl;
  }
}

}  // namespace rvslam
#endif  // _RVSLAM_TIME_H_

