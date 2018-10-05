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
#define DEBUG_SEGMENT_GROW

// #define DEBUG_SEGMENT_MERGE
// #define DEBUG_SEGMENT_MERGE_STEP2

// #define DEBUG_FINAL_TRIMAP

// output information on console.
#define OPEN_DEBUG 0


// Global Data Type and Structure.
#define pass (void)0
typedef unsigned char  UINT8;
typedef unsigned short UINT16;
typedef unsigned int   UINT32;
typedef uint64_t       UINT64;

typedef char           SINT8;
typedef short          SINT16;
typedef int32_t        SINT32;
typedef int64_t        SINT64;

typedef void           VOID;
typedef bool           BOOL;

//
typedef struct SeedNode_OneKey{
    int id0;
    double cost;
    
    SeedNode_OneKey(int a=0, double c=0){id0=a; cost=c;}
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
    int id0;
    int id1;
    double cost;

    int tmp0;
    int tmp1;
    SeedNode(int a=0, int b=0, double c=0, int d=0, int e=0){id0=a; id1=b; cost=c; tmp0=d; tmp1=e;}
}Seed;
struct SeedCmp{
    bool operator()(const Seed &lhs, const Seed &rhs){
        return lhs.cost > rhs.cost;
    }
};

//
typedef struct MapKey_2D{
    int id0;
    int id1;
    MapKey_2D(int a=0, int b=0){
        id0 = a < b? a : b; 
        id1 = a < b? b : a;
    }
}Mkey_2D;
struct MKey2DCmp{
    bool operator()(const Mkey_2D &lhs, const Mkey_2D &rhs){
        if(lhs.id0 != rhs.id0)
            return lhs.id0 < rhs.id0;
        else
            return lhs.id1 < rhs.id1;
    }
};

//
typedef struct MapKey_3D{
    int id0;
    int id1;
    int id2;
    MapKey_3D(int a=0, int b=0, int c=0){id0=a; id1 = b; id2=c;}
}Mkey_3D;
struct MKey3DCmp{
    bool operator()(const Mkey_3D &lhs, const Mkey_3D &rhs){
        if(lhs.id0 != rhs.id0)
            return lhs.id0 < rhs.id0;
        else if(lhs.id1 != rhs.id1)
            return lhs.id1 < rhs.id1;
        else
            return lhs.id2 < rhs.id2;
    }
};

// Global Parameters.
typedef struct Global_Parameters{

    // segment fitting parameters.
    double segFit_dp_semdiff_thr;
    double segFit_dp_bic_alpha;
    double segFit_dp_err_thr;
    double segFit_dp_inf_err;

    // segment growing paramters.
    double  segGrow_seed_sem_alpha;
    double  segGrow_seed_bic_alpha;
    double  segGrow_seed_bic_scale;
    int segGrow_seed_size_thr;
    
    double  segGrow_extd_semdiff_thr;

    double  segGrow_shrk_bic_alpha;
    int segGrow_shrk_bic_addi_len;
    double  segGrow_shrk_fit_cost_thr;
    double  segGrow_shrk_fit_cost_penalty;
    double  segGrow_shrk_cost_thr;

    int segGrow_proposal_size_thr;
    bool   segGrow_rm_label0;

    // segment merge.
    int merge_supix_bic_addi_len;
    
    double  merge_edge_inf_cost;
    double  merge_edge_conn_alpha;
    double  merge_edge_geo_alpha;
    double  merge_edge_bic_alpha;
    double  merge_edge_biascost_alpha;
    double  merge_edge_semdiff_thr;
    double  merge_edge_semcost_thr;
    double  merge_edge_semcost_alpha;

    bool   merge_gen_svm_train_en;
    bool   merge_inst_svm_en;
    bool   merge_merge_svm_en;
    double merge_merge_svm_dec_thr;
    double  merge_merger_thr;

    // tri-map.
    int tri_supix_bic_addi_len;
    double  tri_supix_bic_scale;

    double  tri_seed_sem_alpha;
    double  tri_seed_fit_alpha;
    double  tri_seed_geo_alpha;
   
    double  tri_edge_fit_alpha;
    double  tri_edge_semdiff_thr;

    double  tri_notseed_prob_thr;
    double  tri_cluster_supix0_prob;
    double  tri_cluster_prob_thr;


    // constructor to assign value
    Global_Parameters(){
        // segment fitting parameters.
        segFit_dp_semdiff_thr  = 5e-1;
        segFit_dp_bic_alpha    = 5e-2;
        segFit_dp_err_thr      = 1e-1;
        segFit_dp_inf_err      = 1e4;

        // segment growing paramters.
        segGrow_seed_sem_alpha = 5e-2;
        segGrow_seed_bic_alpha = 5e-1;
        segGrow_seed_bic_scale = 1e-1;
        segGrow_seed_size_thr  = 5;

        segGrow_extd_semdiff_thr      = 5e-1;

        segGrow_shrk_bic_alpha        = 5e1;
        segGrow_shrk_bic_addi_len     = 2;
        segGrow_shrk_fit_cost_thr     = 1e-1;
        segGrow_shrk_fit_cost_penalty = 1e4;
        segGrow_shrk_cost_thr         = 0;

        segGrow_proposal_size_thr     = 8;
        segGrow_rm_label0             = true;

        // segment merge.
        merge_supix_bic_addi_len = 1;

        merge_edge_inf_cost       = 1e9;
        merge_edge_conn_alpha     = 1e1;
        merge_edge_geo_alpha      = 1e-2;
        merge_edge_bic_alpha      = 5e-3;
        merge_edge_biascost_alpha = 5e-1;
        merge_edge_semdiff_thr    = 6e-1;
        merge_edge_semcost_thr    = 4e-1;
        merge_edge_semcost_alpha  = 1e-1;

        merge_gen_svm_train_en   = 1;
        merge_inst_svm_en        = 0;
        merge_merge_svm_en       = 0;
        merge_merge_svm_dec_thr  = 0.3;
        merge_merger_thr         = 0;

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
#define HIST_DIST_NUM_BIN 10
double glb_hist_dist_thr[] = {0.1,  0.2,  0.3,  0.4,
                             0.5,  0.6,  0.7,  0.8,  0.9, 1.01};
template<typename T2>
int vote2histogram_distance(T2 val){
    for(int k =0; k<HIST_DIST_NUM_BIN; k++){
        if(val < glb_hist_dist_thr[k])
            return k;
    }
    return HIST_DIST_NUM_BIN-1;
}


template<typename T2>
double _ChiDifference(std::vector<T2> &obsV, std::vector<T2> &expV){
    double diff = 0; 
    for(int k=0; k<obsV.size(); k++){
        diff += double(pow(obsV[k]-expV[k], 2))/double(expV[k]+obsV[k]);
    }

    return diff;
}

template<typename T>
double _NegativeLog(T prob){
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
    stream<<std::setfill('0')<<std::setw(int(fix_len))<<int(val);
    stream>>str;
    stream.clear();
    
    return str;
}

template<typename T>
inline int _String2Int(std::string str){
    int val;
    std::stringstream stream;
    stream << str;
    stream >> val;
    return T(val);
}

#endif
