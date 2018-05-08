// csio_time.h
//
//  Channeled stream IO based on C++ standard library.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _CSIO_TIME_H_
#define _CSIO_TIME_H_

#if defined(WIN32) && !defined(CYGWIN)

#include <windows.h>
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_100NANOSECS 116444736000000000Ui64
#else
#define DELTA_EPOCH_IN_100NANOSECS 116444736000000000ULL
#endif

namespace csio {

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

}  // namespace csio

#else  // defined(WIN32) && !defined(CYGWIN)

#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>

namespace csio {

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

}  // namespace csio

#endif  // defined(WIN32) && !defined(CYGWIN)

#endif  // _CSIO_TIME_H_

