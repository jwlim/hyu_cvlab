// spm2csio.cc
//
// SPM-to-CSIO converter with on-the-fly conversion through std streams.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)

#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "csio_stream.h"

//#include "HSD.h"
//#include "HStream.h"
//#include "HVideoUtil.h"
//#include "HTimeUtil.h"
//#include "HStrUtil.h"
//#include "HThread.h"

//#if (defined(WIN32) || defined(WIN64)) && !defined(CYGWIN)
//#include <fcntl.h>
//#include <io.h>
//#endif

using namespace std;

#define HDRBLKSZ 1024

DEFINE_string(in, "-", "Input SPM file path (- for stdin).");
DEFINE_string(out, "-", "Output CSIO filepath (- for stdout).");

//-----------------------------------------------------------------------------

struct SPM {
  struct Frame {
    unsigned long buflen;
    string type, desc;
    char* bufptr;
    double* tsptr;
  };
  FILE* fp;
  vector<Frame> frames;
  vector<char> buf;
  long frmno;
};

bool OpenSPM(const string& filepath, SPM* spm) {
  FILE *fp = NULL;
  if (filepath == "-") {  // stdin
    fp = stdin;
//#if (defined(WIN32) || defined(WIN64)) && !defined(CYGWIN)
//    _setmode(_fileno(fp), _O_BINARY);
//#endif
  } else if ((fp = fopen(filepath.c_str(), "rb")) == NULL) {
    LOG(ERROR) << "failed to open '" << filepath << "'.";
    return false;
  }

  // Read the header of the SPM file : read the first string to check if
  // it is in the old format.
  char header[HDRBLKSZ + 1];
  if (fgets(header, HDRBLKSZ, fp) == NULL) {
    fclose(fp);
    LOG(ERROR) << "invalid header (" << filepath << "): " << header;
    return false;
  }
  if (strncmp(header, "spm.sf", 6) != 0) {
    fclose(fp);
    LOG(ERROR) << "legacy spm header (" << filepath << "): " << header;
    return false;
  }

  // Read the rest of the header and parse it.
  int pos = strlen(header);
  fread(&header[pos], 1, HDRBLKSZ - pos, fp);
  header[HDRBLKSZ] = 0;
  VLOG(3) << "header: '" << header << "' " << ftell(fp);

  int numch = 0, shotbufsz = 0;
  sscanf(&header[6], "%d", &numch);
  if (numch <= 0) {
    fclose(fp);
    LOG(ERROR) << "invalid header (" << filepath << "): numch=" << numch;
    return false;
  }

  spm->frames.resize(numch);
  unsigned long bufsize = 0;
  for (int i = 0; i < numch; ++i) {
    SPM::Frame &frm = spm->frames[i];
    sscanf(&header[pos + 1], "%lu", &frm.buflen);
    pos += strlen(&header[pos + 1]) + 1;
    frm.type = string(&header[pos + 1]);
    pos += strlen(&header[pos + 1]) + 1;
    frm.desc = string(&header[pos + 1]);
    pos += strlen(&header[pos + 1]) + 1;
    bufsize += frm.buflen;
  }
  spm->fp = fp;
  spm->buf.resize(bufsize);
  char* bufptr = &spm->buf[0];
  for (int i = 0, idx = 0; i < numch; ++i) {
    SPM::Frame &frm = spm->frames[i];
    frm.bufptr = &spm->buf[idx];
    idx += frm.buflen;
    frm.buflen -= sizeof(double);
    frm.tsptr = reinterpret_cast<double*>(frm.bufptr + frm.buflen);
  }
  spm->frmno = 0;
  return true;
}

bool FetchSPM(SPM* spm) {
  FILE* fp = spm->fp;
  vector<char>& buf = spm->buf;
  if (fp == NULL || feof(fp) || fread(&buf[0], 1, buf.size(), fp) == 0) {
    LOG(ERROR) << "failed to read frame " << spm->frmno;
    return false;
  }
  ++spm->frmno;
  return true;
}

void CloseSPM(SPM* spm) {
  if (spm->fp && spm->fp != stdin) fclose(spm->fp);
  spm->fp = NULL;
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "starting up...";

  // Setup SPM input stream.
  LOG(INFO) << "setting up SPM (in=" << FLAGS_in << ").";
  SPM spm;
  if (OpenSPM(FLAGS_in, &spm) == false) {
    LOG(ERROR) << "failed to open SPM (in=" << FLAGS_in << ").";
    return -1;
  }
  // Setup csio::OutputStream.
  LOG(INFO) << "setting up csio::Outputstream (out=" << FLAGS_out << ").";
  vector<csio::ChannelInfo> channels(spm.frames.size());
  for (int i = 0; i < spm.frames.size(); ++i) {
    const SPM::Frame &frm = spm.frames[i];
    csio::ChannelInfo& ch = channels[i];
    ch.id = i;
    string type, pixel, size, maxv;
    const char* delim = " \t:<";
    size_t pos = csio::GetToken(frm.type, 0, &type, delim);
    if (type == "img") {
      pos = csio::GetToken(frm.type, pos, &pixel, delim);
      pos = csio::GetToken(frm.type, pos, &size, delim);
      pos = csio::GetToken(frm.type, pos, &maxv, delim);
      if (pixel == "uchar3") pixel = "rgb8";
      ch.type = "image/x-csio-raw;pixel=" + pixel + ";size=" + size;
      if (!maxv.empty()) ch.type += ";maxv=" + maxv;
      LOG(INFO) << "img channel type '" << ch.type << "'.";
    } else {
      LOG(ERROR) << "unknown channel type '" << type << "'.";
      ch.type = frm.type;  // TODO: Needs conversion.
    }
  }
  map<string, string> config;
  csio::OutputStream csout;
  if (csout.Setup(channels, config, FLAGS_out) == false) {
    CloseSPM(&spm);
    LOG(ERROR) << "failed to open csio::OutputStream (in=" << FLAGS_in << ").";
    return -1;
  }
  // Pipe the contents from input to output.
  LOG(INFO) << "start streaming.";
  while (FetchSPM(&spm)) {
    for (int i = 0; i < spm.frames.size(); ++i) {
      const SPM::Frame &frm = spm.frames[i];
      csout.Push(i, frm.bufptr, frm.buflen, *frm.tsptr);
    }
  }
  CloseSPM(&spm);
  return 0;
}

//-----------------------------------------------------------------------------

