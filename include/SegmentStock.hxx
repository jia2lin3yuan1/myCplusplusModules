#ifndef _SEGMENT_STOCK_HXX
#define _SEGMENT_STOCK_HXX

#include "utils/read_write_img.hxx"
using namespace std;

typedef struct SegmentInformation{
    float fit_err;
    vector<float> sem_score;
}SegInfo;

typedef struct All_Segment{
    UINT32 y0, x0; //[pt0, pt1]
    UINT32 y1, x1;
    All_Segment(UINT32 a, UINT32 b, UINT32 c, UINT32 d){
        y0 = a;  x0 = b;  
        y1 = c;  x1 = d;
    } 
} Aseg; //[st, end)
struct AsegCmp{
    bool operator()(const Aseg& lhs, const Aseg& rhs)const{
        if(lhs.y0 != rhs.y0)
            return lhs.y0 < rhs.y0;
        else if(lhs.x0 != rhs.x0)
            return lhs.x0 < rhs.x0;
        else if(lhs.y1 != rhs.y1)
            return lhs.y1 < rhs.y1;
        else
            return lhs.x1 < rhs.x1;
    }
};

typedef struct DP_Segment{
    UINT32 len;
    UINT32 y0, x0; // [pt0, pt1]
    UINT32 y1, x1;
    SegInfo seg_info;
    DP_Segment(){
        len=0; y0=0; x0=0; y1=0; x1=0;
    }
    DP_Segment(UINT32 z, UINT32 a, UINT32 b, UINT32 c, UINT32 d){
        len = z;
        y0  = a;  x0 = b;  
        y1  = c;  x1 = d;
    } 
} DpSeg; // [pt0, pt1)


enum SegType {e_seg_h=0, e_seg_v,  e_seg_type_num};

/*
 * Class: SegmentStock.
 * There are two kind of segments in the stock:
 * 1. all_segment: possible segment locates between any two key points.
 * 2. dp_segment:  optimal segment comes from Dynamic Programming Partition in SegmentFitting.
 */
class Segment_Stock{

protected:
    // Variables.
    map<Aseg, float, AsegCmp> m_all_seg;
    map<UINT32, DpSeg>  m_dp_seg;

    UINT32 m_ht, m_wd;
    CDataTempl<UINT32> m_dp_segInfo;

    // Parameter.

public: //Functions
    Segment_Stock(UINT32 ht, UINT32 wd){
        m_ht = ht;   m_wd = wd;
        m_dp_segInfo.Init(ht, wd, e_seg_type_num);
    }

    CDataTempl<UINT32> &GetSegmentLabelImage(){
        return m_dp_segInfo;
    }

    void AssignAllSegments(CDataTempl<float> &seg_info, vector<UINT32> &all_idxs, vector<UINT32> &ptY, vector<UINT32> &ptX);
    void AssignDpSegments(CDataTempl<float> &seg_info, auto &semScore, vector<UINT32> &dp_idxs, vector<UINT32> &ptY, vector<UINT32> &ptX);
   
    float GetAllSegFitError(UINT32 y0, UINT32 x0, UINT32 y1, UINT32 x1);
    
    void GetDpSegmentById(DpSeg &dp_seg, UINT32 id);
    void GetDpSegmentByCoord(DpSeg &dp_seg, UINT32 py, UINT32 px, SegType s_type);
    UINT32 GetDpSegmentSize();
    
};


void Segment_Stock::AssignAllSegments(CDataTempl<float> &seg_info,  vector<UINT32> &all_idxs, vector<UINT32> &ptY, vector<UINT32> &ptX){
    // assign segments.
    for(int k0=0; k0<all_idxs.size()-1; k0++){
        UINT32 y0 = ptY[all_idxs[k0]];
        UINT32 x0 = ptX[all_idxs[k0]];
       
        for(int k1=k0+1; k1 < all_idxs.size(); k1++){
            UINT32 y1 = ptY[all_idxs[k1]];
            UINT32 x1 = ptX[all_idxs[k1]];
           
            Aseg a_seg(y0, x0, y1, x1);
            m_all_seg[a_seg] = seg_info.GetData(k0, k1);
        }
    }
}

void Segment_Stock::AssignDpSegments(CDataTempl<float> &seg_info, auto &semScore, vector<UINT32> &dp_idxs, vector<UINT32> &ptY, vector<UINT32> &ptX){
    UINT32 seg_id = m_dp_seg.size();
    
    // assign. 
    UINT32 y0 = ptY[dp_idxs[1]];
    UINT32 x0 = ptX[dp_idxs[1]];
    for(UINT32 ck=3; ck<dp_idxs.size(); ck+=2){
        UINT32 y1 = ptY[dp_idxs[ck]];
        UINT32 x1 = ptX[dp_idxs[ck]];
        UINT32 seg_len = dp_idxs[ck] - dp_idxs[ck-2]; 
        
        DpSeg dp_seg(seg_len, y0, x0, y1, x1);
        dp_seg.seg_info.fit_err = seg_info.GetData(dp_idxs[ck-3], dp_idxs[ck-1]);    
        Mkey_2D key(dp_idxs[ck-3], dp_idxs[ck-1]);
        dp_seg.seg_info.sem_score.assign(semScore[key].begin(), semScore[key].end());
        m_dp_seg[seg_id] = dp_seg;
      
        // record seg id to image.
        if(y1==y0){
            m_dp_segInfo.ResetBulkData(seg_id, y0, 1, x0, x1-x0, e_seg_h, 1);
#ifdef DEBUG_SEGMENT_STOCK
            m_dp_segInfo.ResetBulkData(x0, y0, 1, x0, x1-x0, e_seg_h, 1);
#endif
        }
        else{
            m_dp_segInfo.ResetBulkData(seg_id, y0, y1-y0, x0, 1, e_seg_v, 1);
#ifdef DEBUG_SEGMENT_STOCK
            m_dp_segInfo.ResetBulkData(y0, y0, y1-y0, x0, 1, e_seg_v, 1);
#endif
        }

        // update for next loop.
        seg_id += 1;
        y0 = y1;   x0 = x1;
    }
}


float Segment_Stock::GetAllSegFitError(UINT32 y0, UINT32 x0, UINT32 y1, UINT32 x1){
    Aseg a_seg(y0, x0, y1, x1);
    return m_all_seg[a_seg];
}

void Segment_Stock::GetDpSegmentById(DpSeg &dp_seg, UINT32 id){
    dp_seg = m_dp_seg[id];
}

void Segment_Stock::GetDpSegmentByCoord(DpSeg &dp_seg, UINT32 py, UINT32 px, SegType s_type){
    UINT32 id = m_dp_segInfo.GetData(py, px, s_type);
    this->GetDpSegmentById(dp_seg, id);
}


UINT32 Segment_Stock::GetDpSegmentSize(){
    return m_dp_seg.size();
}
#endif
