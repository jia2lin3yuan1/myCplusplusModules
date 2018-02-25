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
typedef struct Global_Parameters{

    // segment fitting parameters.
    float segFit_bic_alpha;
    float segFit_err_thr;
    float segFit_inf_err;

    // segment growing paramters.
    float segGrow_seed_bic_alpha;
    float segGrow_seed_bic_scale;
    UINT32 segGrow_seed_size_thr;

    float segGrow_shrk_bic_alpha;
    UINT32 segGrow_shrk_bic_addi_len;
    float segGrow_shrk_fit_cost_thr;
    float segGrow_shrk_fit_cost_penalty;
    float segGrow_shrk_cost_thr;

    UINT32 segGrow_proposal_size_thr;

    // segment merge.
    
    Global_Parameters(){
        // segment fitting parameters.
        segFit_bic_alpha = 8e-2;
        segFit_err_thr   = 5e-2;
        segFit_inf_err   = 1e3;

        // segment growing paramters.
        segGrow_seed_bic_alpha = 5e-2;
        segGrow_seed_bic_scale = 5e-1;
        segGrow_seed_size_thr  = 5;

        segGrow_shrk_bic_alpha    = 1;
        segGrow_shrk_bic_addi_len = 2;
        segGrow_shrk_fit_cost_thr = 4e-2;
        segGrow_shrk_fit_cost_penalty = 1e3;
        segGrow_shrk_cost_thr      = 0;

        UINT32 segGrow_proposal_size_thr = 40;

        // segment merge.
    }

}GlbParam;












// Global Functions.
template<typename T>
inline T _CLIP(T v, T minV, T maxV){
    return v<minV? minV : (v > maxV? maxV : v);
}


template<typename T>
inline std::string _Int2String(T val, T fix_len=5){
    std::string str;
    std::stringstream stream;
    stream<<std::setfill('0')<<std::setw(UINT32(fix_len))<<UINT32(val);
    stream>>str;
    stream.clear();
    
    return str;
}

template<typename T>
inline UINT32 _String2Int(std::string str){
    UINT32 val;
    std::stringstream stream;
    stream << str;
    stream >> val;
    return T(val);
}

#endif
