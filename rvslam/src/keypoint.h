
#ifndef __INCLUDE_KEYPOINT_H__
#define __INCLUDE_KEYPOINT_H__
#include <iostream>
using namespace std;

template<typename Type_> static inline Type_ saturate_cast(unsigned char v) { return Type_(v); }
template<typename Type_> static inline Type_ saturate_cast(int v) { return Type_(v); }
template<typename Type_> static inline Type_ saturate_cast(float v) { return Type_(v); }

template<typename Type_> class Ptr_{
public:
	typedef Type_ value_type;
	Ptr_() {};
	Ptr_(Type_ _x, Type_ _y) : x(_x), y(_y) {}
	Type_ x, y; // the point coordinates
};

typedef Ptr_<int> Ptr2i;
typedef Ptr_<float> Ptr2f;
typedef Ptr_<double> Ptr2d;

#endif