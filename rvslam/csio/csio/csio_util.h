// csio_util.h
//
//  Channeled stream IO based on C++ standard library.
//
// Authors: Jongwoo Lim (jongwoo.lim@gmail.com)
//

#ifndef _CSIO_UTIL_H_
#define _CSIO_UTIL_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <vector>
#include <sstream>

namespace csio {

inline std::string FormatStr(const std::string fmt, ...) {
  char buf[1024];
  va_list a;
  va_start(a, fmt);
  vsprintf(buf, fmt.c_str(), a);
  va_end(a);
  return std::string(buf);
}

inline bool Contains(const std::string& str, const std::string& substr) {
  return (str.find(substr) != std::string::npos);
}

inline bool ContainsAnyOf(const std::string& str, const std::string& chars) {
  return (str.find_first_of(chars) != std::string::npos);
}

inline std::string StrTok(const std::string& delim, std::string* str) {
  size_t i = str->find_first_not_of(delim), j;
  std::string tok;
  if (i == std::string::npos) return tok;
  if ((j = str->find_first_of(delim, i)) == std::string::npos) {
    tok = str->substr(i);
    str->clear();
  } else {
    tok = str->substr(i, j - i);
    *str = str->substr(j + 1);
  }
  return tok;
}

inline size_t SplitStr(const std::string& str, const std::string& delim,
                       std::vector<std::string>* tokens) {
  tokens->clear();
  size_t i, j, npos = std::string::npos;
  for (i = 0; (i = str.find_first_not_of(delim, i)) != npos; i = j) {
    j = str.find_first_of(delim, i);
    if (j == npos) tokens->push_back(str.substr(i));
    else tokens->push_back(str.substr(i, j - i));
  }
  return tokens->size();
}

inline size_t SplitStr(const std::string& str, const std::string& delim,
                       const std::string& quote,
                       std::vector<std::string>* tokens) {
  tokens->clear();
  size_t i, j, npos = std::string::npos;
  std::string dq = delim + quote;
  for (i = 0; i != npos && (i = str.find_first_not_of(delim, i)) != npos;
       i = j) {
    char ch = str[i];
    if (quote.find(ch) == npos) j = str.find_first_of(dq, i), ch = 0;
    else j = str.find((ch == '(') ? ')' :
                      (ch == '[' || ch == '{' || ch == '<') ? ch + 2 : ch, ++i);
    tokens->push_back(str.substr(i, (j == npos)? j : j-i));
    if (ch != 0 && j != npos) j++;
  }
  return tokens->size();
}

inline size_t SplitStr(const std::string& str, const std::string& delim,
                       const std::string& assign,
                       std::map<std::string, std::string>* dict_ptr) {
  std::map<std::string, std::string>& dict = *dict_ptr;
  dict.clear();
  size_t i, j, k, npos = std::string::npos;
  for (i = 0; (i = str.find_first_not_of(delim, i)) != npos; i = j) {
    j = str.find_first_of(delim, i);
    const std::string tok = (j == npos) ? str.substr(i) : str.substr(i, j - i);
    if ((k = tok.find_first_of(assign)) == npos) dict[tok] = "";
    else dict[tok.substr(0, k)] = tok.substr(k + 1);
  }
  return dict.size();
}

inline size_t SplitStr(const std::string& str, const std::string& delim,
                       const std::string& quote, const std::string& assign,
                       std::map<std::string, std::string>* dict_ptr) {
  std::map<std::string, std::string>& dict = *dict_ptr;
  dict.clear();
  std::string dqa = delim + quote + assign;
  size_t i, j, npos = std::string::npos;
  for (i = 0; i != npos && (i=str.find_first_not_of(delim,i)) != npos; i = j) {
    char c = str[i];
    if (quote.find(c) == npos) j = str.find_first_of(dqa, i), c = 0;
    else j = str.find((c == '(') ? ')' :
                      (c == '[' || c == '{' || c == '<') ? c + 2 : c, ++i);
    const std::string key = str.substr(i, (j == npos) ? j : j - i);
    j = str.find_first_not_of(delim, (c != 0 && j != npos) ? j + 1 : j);
    if (j == npos || assign.find(str[j]) == npos) {
      dict[key] = "";
    } else if ((j = str.find_first_not_of(delim, j + 1)) == npos) {
      dict[key] = "";
    } else {
      c = str[i = j];
      if (quote.find(c) == npos) j = str.find_first_of(dqa, i), c = 0;
      else j = str.find((c == '(') ? ')':
                        (c == '[' || c == '{' || c == '<') ? c + 2 : c, ++i);
      dict[key] = str.substr(i, (j == npos) ? j : j - i);
      if (c != 0 && j != npos) j++;
    }
  }
  return dict.size();
}

}  // namespace csio
#endif  // _CSIO_UTIL_H_
