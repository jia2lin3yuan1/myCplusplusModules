#ifndef PLATFORM_DEFS_HXX
#define PLATFORM_DEFS_HXX

#include <stdint.h>
#include <iostream>
#include <fstream>
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

// #define DEBUG_FINAL_TRIMAP

#define OPEN_DEBUG 0 // output information on console.


// Global Data Type and Structure.
#define pass (void)0
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
typedef struct SeedNode_OneKey{
    UINT32 id0;
    float cost;
    
    SeedNode_OneKey(UINT32 a=0, float c=0){id0=a; cost=c;}
}Seed_1D;

struct SeedCmp_1D{ // Increasing order
    bool operator()(const Seed_1D &lhs, const Seed_1D &rhs){
        return lhs.cost > rhs.cost;
    }
};
struct SeedCmp_1D_Dec{ // Decreasing order.
    bool operator()(const Seed_1D &lhs, const Seed_1D &rhs){
        return lhs.cost < rhs.cost;
    }
};

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
    float segFit_dp_semdiff_thr;
    float segFit_dp_bic_alpha;
    float segFit_dp_err_thr;
    float segFit_dp_inf_err;

    // segment growing paramters.
    float  segGrow_seed_sem_alpha;
    float  segGrow_seed_bic_alpha;
    float  segGrow_seed_bic_scale;
    UINT32 segGrow_seed_size_thr;
    
    float  segGrow_extd_semdiff_thr;

    float  segGrow_shrk_bic_alpha;
    UINT32 segGrow_shrk_bic_addi_len;
    float  segGrow_shrk_fit_cost_thr;
    float  segGrow_shrk_fit_cost_penalty;
    float  segGrow_shrk_cost_thr;

    UINT32 segGrow_proposal_size_thr;
    bool   segGrow_rm_label0;

    // segment merge.
    UINT32 merge_supix_bic_addi_len;
    
    float  merge_edge_conn_alpha;
    float  merge_edge_geo_alpha;
    float  merge_edge_bic_alpha;
    float  merge_edge_semdiff_thr;
    float  merge_edge_semdiff_pnty;

    float  merge_merger_thr;

    // tri-map.
    UINT32 tri_supix_bic_addi_len;
    float  tri_supix_bic_scale;

    float  tri_seed_sem_alpha;
    float  tri_seed_fit_alpha;
    float  tri_seed_geo_alpha;
   
    float  tri_edge_fit_alpha;
    float  tri_edge_semdiff_thr;

    float  tri_notseed_prob_thr;
    float  tri_cluster_supix0_prob;
    float  tri_cluster_prob_thr;


    // constructor to assign value
    Global_Parameters(){
        // segment fitting parameters.
        segFit_dp_semdiff_thr  = 7e-1;
        segFit_dp_bic_alpha    = 1e-1;
        segFit_dp_err_thr      = 5e-2;
        segFit_dp_inf_err      = 1e4;

        // segment growing paramters.
        segGrow_seed_sem_alpha = 5e-2;
        segGrow_seed_bic_alpha = 5e-1;
        segGrow_seed_bic_scale = 5e-1;
        segGrow_seed_size_thr  = 5;

        segGrow_extd_semdiff_thr      = 7e-1;

        segGrow_shrk_bic_alpha        = 5e-1;
        segGrow_shrk_bic_addi_len     = 2;
        segGrow_shrk_fit_cost_thr     = 5e-2;
        segGrow_shrk_fit_cost_penalty = 1e4;
        segGrow_shrk_cost_thr         = 0;

        segGrow_proposal_size_thr     = 40;
        segGrow_rm_label0             = true;

        // segment merge.
        merge_supix_bic_addi_len = 1;
        
        merge_edge_conn_alpha    = 1e1;
        merge_edge_geo_alpha     = 2e2;
        merge_edge_bic_alpha     = 1e0;
        merge_edge_semdiff_thr   = 7e-1;
        merge_edge_semdiff_pnty  = 1e9;

        merge_merger_thr         = 1e4;

        // tri-map generate
        tri_supix_bic_addi_len   = 1;
        tri_supix_bic_scale      = 1;
        
        tri_seed_sem_alpha       = 0;//1e-2;
        tri_seed_fit_alpha       = 0;
        tri_seed_geo_alpha       = 5e-1;
    
        tri_edge_fit_alpha       = 5e-1;
        tri_edge_semdiff_thr     = 1e0;
        
        tri_notseed_prob_thr     = 6e-1;
        tri_cluster_prob_thr     = 5e-1;
        tri_cluster_supix0_prob  = 5e-1;
    }

}GlbParam;




// Global Functions.
#define HIST_W_NUM_BIN 12
float glb_hist_w_thr[] = {-5e-2, -2e-2, -1.5e-2, -1e-2, -5e-3, 0, 5e-3, 1e-2, 1.5e-2, 2e-2, 5e-2};
template<typename T2>
UINT32 vote2histogram_w(T2 val){
    for(int k =0; k<HIST_W_NUM_BIN-1; k++){
        if(val < glb_hist_w_thr[k])
            return k;
    }
    return HIST_W_NUM_BIN-1;
}


template<typename T2>
float _ChiDifference(std::vector<T2> &obsV, std::vector<T2> &expV){
    float diff = 0; 
    for(UINT32 k=0; k<obsV.size(); k++){
        diff += float(pow(obsV[k]-expV[k], 2))/float(expV[k]+obsV[k]);
    }

    return diff;
}

template<typename T>
float _NegativeLog(T prob){
    return -log(prob/(1-prob+1e-5));
}

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
