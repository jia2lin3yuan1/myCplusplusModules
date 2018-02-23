#ifndef PLATFORM_DEFS_HXX
#define PLATFORM_DEFS_HXX

#include <stdint.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <string>
#include<iterator>
#include <map>
#include <vector>
#include <queue>
#include <stack>

#include <math.h>

typedef unsigned char  UINT8;
typedef char           SINT8;
typedef unsigned short UINT16;
typedef short          SINT16;
typedef uint32_t       UINT32;
typedef int32_t        SINT32;
typedef uint64_t       UINT64;
typedef int64_t        SINT64;
typedef void           VOID;
typedef bool           BOOL;


// Global Parameters.
#define SEG_FIT_BIC_ALPHA 8e-2
#define SEG_FIT_ERR_THR   4e-2
#define SEG_FIT_INF_ERR   1e3

#define SEG_GROW_SEED_BIC_ALPHA 1e-2
#define SEG_GROW_SEED_BIC_SCALE 2e-1
#define SEG_GROW_RM_BIC_ALPHA  1
#define SEG_GROW_RM_COST  0


// Global Functions.
inline int _CLIP(int v, int minV, int maxV){
    return v<minV? minV : (v > maxV? maxV : v);
}
inline std::string _Int2String(int val, int fix_len=5){
    std::string str;
    std::stringstream stream;
    stream<<std::setfill('0')<<std::setw(fix_len)<<val;
    stream>>str;
    stream.clear();
    
    return str;
}
inline int _String2Int(std::string str){
    int val;
    std::stringstream stream;
    stream << str;
    stream >> val;
    return val;
}

#endif
