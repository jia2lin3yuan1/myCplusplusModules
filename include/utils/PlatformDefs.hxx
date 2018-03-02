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

/** Turn on / Turn off some functions. **/
// #define DEBUG_SEGMENT_STOCK

// #define DEBUG_SEGMENT_GROW_STEP

// #define DEBUG_SEGMENT_GROW

// #define DEBUG_SEGMENT_MERGE

#define OPEN_DEBUG 0


// Global Data Type and Structure.
typedef unsigned char  UINT8;
typedef char           SINT8;
typedef unsigned short UINT16;
typedef short          SINT16;
typedef unsigned int   UINT32;
typedef int32_t        SINT32;
typedef uint64_t       UINT64;
typedef int64_t        SINT64;
typedef void           VOID;
typedef bool           BOOL;

//
typedef struct SeedNode{
    UINT32 id0;
    UINT32 id1;
    float cost;

    UINT32 tmp0;
    UINT32 tmp1;
    SeedNode(UINT32 a=0, UINT32 b=0, float c=0, UINT32 d=0, UINT32 e=0){id0=a; id1=b; cost=c; tmp0=d; tmp1=e;}
}Seed;
struct SeedCmp{
    bool operator()(const Seed &lhs, const Seed &rhs){
        return lhs.cost > rhs.cost;
    }
};

//
typedef struct MapKey_2D{
    UINT32 id0;
    UINT32 id1;
    MapKey_2D(UINT32 a=0, UINT32 b=0){id0=a; id1 = b;}
}Mkey_2D;
struct MKey2DCmp{
    bool operator()(const Mkey_2D &lhs, const Mkey_2D &rhs){
        if(lhs.id0 != rhs.id0)
            return lhs.id0 < rhs.id0;
        else
            return lhs.id1 < rhs.id1;
    }
};


// Global Parameters.
typedef struct Global_Parameters{

    // segment fitting parameters.
    float segFit_bic_alpha;
    float segFit_err_thr;
    float segFit_inf_err;

    // segment growing paramters.
    float  segGrow_seed_bic_alpha;
    float  segGrow_seed_bic_scale;
    UINT32 segGrow_seed_size_thr;

    float  segGrow_shrk_bic_alpha;
    UINT32 segGrow_shrk_bic_addi_len;
    float segGrow_shrk_fit_cost_thr;
    float segGrow_shrk_fit_cost_penalty;
    float segGrow_shrk_cost_thr;

    UINT32 segGrow_proposal_size_thr;

    // segment merge.
    float  merge_supix_bic_alpha;
    UINT32 merge_supix_bic_addi_len;
    float  merge_merger_thr;
    
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

        segGrow_proposal_size_thr = 40;

        // segment merge.
        merge_supix_bic_alpha    = 2e-1;
        merge_supix_bic_addi_len = 1;
        merge_merger_thr         = 0;
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
