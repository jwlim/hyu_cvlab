// csio_test.cc
//

#include <fstream>
#include <sstream>
//#include <algorithm>

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "csio_stream.h"

using namespace std;

TEST(CSIO, CSIOTest) {
}

TEST(CSIO, CSIOBasicDynamicCastTest) {
  ios* p = &cout;
  EXPECT_EQ(NULL, dynamic_cast<ofstream*>(p));
  ofstream fos;
  p = &fos;
  EXPECT_TRUE(dynamic_cast<ofstream*>(p) != NULL);
}

TEST(CSIO, CSIOInputOutputStreamTest) {
  csio::OutputStream csio_os;
  csio::Frame frame;
  EXPECT_FALSE(csio_os.Push(frame));

  vector<csio::ChannelInfo> channels;
  channels.push_back(csio::ChannelInfo(0, "test_channel1", "a huge stream'_'"));
  channels.push_back(csio::ChannelInfo(2, "test channel 2", "little_channel"));
  map<string, string> config;
  EXPECT_TRUE(csio_os.Setup(channels, config, cerr));
  EXPECT_TRUE(csio_os.PushStr(0, "test data 0 to ch0          "));
  EXPECT_TRUE(csio_os.PushStr(2, "test data 0 to ch2"));
  EXPECT_TRUE(csio_os.PushStr(0, "test data 1 to ch0     "));
  csio_os.Close();

  stringstream ss;
  csio_os.Setup(channels, config, ss);
  EXPECT_TRUE(csio_os.PushStr(0, "test data 0 to ch0          "));
  EXPECT_TRUE(csio_os.PushStr(2, "test data 0 to ch2"));

  csio::InputStream csio_is;
  EXPECT_FALSE(csio_is.Fetch(&frame));
  EXPECT_TRUE(csio_is.Setup(ss));
  EXPECT_EQ(channels.size(), csio_is.channels().size());
  for (int i = 0; i < channels.size(); ++i) {
    EXPECT_EQ(channels[i].id, csio_is.channels()[i].id);
    EXPECT_EQ(channels[i].type, csio_is.channels()[i].type);
    EXPECT_EQ(channels[i].desc, csio_is.channels()[i].desc);
  }

  EXPECT_TRUE(csio_is.Fetch(&frame));
  EXPECT_EQ(0, frame.chid);
  EXPECT_TRUE(csio_is.Fetch(&frame));
  EXPECT_EQ(2, frame.chid);

  vector<char> large_buf(1024 * 1024, ' ');
  EXPECT_TRUE(csio_os.Push(0, large_buf));

  EXPECT_TRUE(csio_is.Fetch(&frame));
  EXPECT_EQ(0, frame.chid);
  EXPECT_EQ(large_buf.size(), frame.buf.size());

  csio_is.Close();
  csio_os.Close();
}

TEST(CSIO, CSIOEscapeStrTest) {
  EXPECT_EQ("''", csio::EscapeStr(""));
  EXPECT_EQ("abc", csio::EscapeStr("abc"));
  EXPECT_EQ("ab\\c", csio::EscapeStr("ab\\c"));
  EXPECT_EQ("'abc def'", csio::EscapeStr("abc def"));
  EXPECT_EQ("'abc def\tghi'", csio::EscapeStr("abc def\tghi"));
  EXPECT_EQ("\"abcdefghi\"", csio::EscapeStr("\"abcdefghi\""));
  EXPECT_EQ("'abc\\' def\t\\'ghi'", csio::EscapeStr("abc' def\t'ghi"));
  EXPECT_EQ("'\\'abc def\tghi\\''", csio::EscapeStr("'abc def\tghi'"));
  EXPECT_EQ("'\\'ab\\\\cd\\\\\\\\ef\\''", csio::EscapeStr("'ab\\cd\\\\ef'"));

  EXPECT_EQ("", csio::UnescapeStr("''"));
  EXPECT_EQ("abc", csio::UnescapeStr("abc"));
  EXPECT_EQ("ab\\c", csio::UnescapeStr("ab\\c"));
  EXPECT_EQ("abc def", csio::UnescapeStr("'abc def'"));
  EXPECT_EQ("abc def\tghi", csio::UnescapeStr("'abc def\tghi'"));
  EXPECT_EQ("\"abcdefghi\"", csio::UnescapeStr("\"abcdefghi\""));
  EXPECT_EQ("abc' def\t'ghi", csio::UnescapeStr("'abc\\' def\t\\'ghi'"));
  EXPECT_EQ("'abc def\tghi'", csio::UnescapeStr("'\\'abc def\tghi\\''"));
  EXPECT_EQ("'ab\\cd\\\\ef'", csio::UnescapeStr("'\\'ab\\\\cd\\\\\\\\ef\\''"));
}

TEST(CSIO, CSIOGetTokenTest) {
  string str = "hello 'world'\t 'a\\'b\\\\'  'c'  ''";
  string tok;
  size_t pos = csio::GetToken(str, 0, &tok);
  EXPECT_EQ("hello", tok);
  pos = csio::GetToken(str, pos, &tok);
  EXPECT_EQ("'world'", tok);
  pos = csio::GetToken(str, pos, &tok);
  EXPECT_EQ("'a\\'b\\\\'", tok);
  pos = csio::GetToken(str, pos, &tok);
  EXPECT_EQ("'c'", tok);
  pos = csio::GetToken(str, pos, &tok);
  EXPECT_EQ("''", tok);
  EXPECT_EQ(string::npos, pos);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}

