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
// output information on console.
#define OPEN_DEBUG 1
#define DEBUG_MERGE_STEP


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

// Global structure
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

    int     merge_bic_addi_len;
    
    double  merge_edge_grad_max;
    double  merge_supix_diff_max;
    double  merge_supix_size_alpha;
    double  merge_supix_unconnect_penalty;

    double merge_supix_rm_mean_thrH;
    double merge_supix_rm_a2p_thrL; // area-to-perimeter
    
    double  merge_merge_thrH;


    // constructor to assign value
    Global_Parameters(double supix_obj_cen_maxV = 20.0){
        merge_bic_addi_len        = 1;
        
        merge_edge_grad_max           = 1.0;
        merge_supix_diff_max          = 1.0;
        merge_supix_size_alpha        = 0;
        merge_supix_unconnect_penalty = 0.1;

        merge_supix_rm_mean_thrH  = (supix_obj_cen_maxV + 1.0);
        merge_supix_rm_a2p_thrL   = 5.0;
        
        merge_merge_thrH          = 100.0;
    }

}GlbParam;


// Global Functions.

#define _MAX2_(x, y) (x)<(y)?(y):(x)
#define _MIN2_(x, y) (x)<(y)?(x):(y)

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
