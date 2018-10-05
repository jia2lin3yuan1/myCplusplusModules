#ifndef _SEGMENT_STOCK_HXX
#define _SEGMENT_STOCK_HXX

#include "utils/read_write_img.hxx"
using namespace std;

typedef struct SegmentFitResult{
    double  fit_err;
    double  w[2];
    double  b[2];
    int ch[2];
    SegmentFitResult(){
        fit_err = 0;
        w[0] = 0;  w[1] = 0;
        b[0] = 0;  b[1] = 0;
        ch[0] = 0; ch[1] = 0;
    }
}SegFitRst;

typedef struct SegmentInformation{
    double fit_err;
    vector<double> sem_score;
}SegInfo;

typedef struct All_Segment{
    int y0, x0; //[pt0, pt1]
    int y1, x1;
    All_Segment(int a, int b, int c, int d){
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
    int len;
    int y0, x0; // [pt0, pt1]
    int y1, x1;
    SegInfo seg_info;
    DP_Segment(){
        len=0; y0=0; x0=0; y1=0; x1=0;
    }
    DP_Segment(int z, int a, int b, int c, int d){
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
    map<Aseg, SegFitRst, AsegCmp> m_all_seg;
    map<int, DpSeg>  m_dp_seg;

    int m_ht, m_wd;
    CDataTempl<int> m_dp_segInfo;
    CDataTempl<int> m_all_segIdx;

    // Parameter.

public: //Functions
    Segment_Stock(int ht, int wd){
        m_ht = ht;   m_wd = wd;
        m_dp_segInfo.Init(ht, wd, e_seg_type_num);
        m_all_segIdx.Init(ht, wd, e_seg_type_num*2);
    }

    CDataTempl<int> &GetSegmentLabelImage(){
        //return m_all_segIdx;
        return m_dp_segInfo;
    }

    void AssignAllSegments(CDataTempl<double> &seg_info, vector<int> &all_idxs, vector<int> &ptY, vector<int> &ptX, int *dist_ch);
    void AssignDpSegments(CDataTempl<double> &seg_info, auto &semScore, vector<int> &dp_idxs, vector<int> &ptY, vector<int> &ptX);
   
    double GetAllSegFitError(int y0, int x0, int y1, int x1);
    double GetAllSegFitErrorOnAny2Points(int y0, int x0, int y1, int x1);
    SegFitRst &GetAllSegFitResultOnAny2Points(int y0, int x0, int y1, int x1); 
    
    void GetDpSegmentById(DpSeg &dp_seg, int id);
    void GetDpSegmentByCoord(DpSeg &dp_seg, int py, int px, SegType s_type);
    int GetDpSegmentSize();
    
};


void Segment_Stock::AssignAllSegments(CDataTempl<double> &seg_info,  vector<int> &all_idxs, vector<int> &ptY, vector<int> &ptX, int *dist_ch){
    
    // assign segments to segment stock
    for(int k0=0; k0<all_idxs.size()-1; k0++){
        int y0 = ptY[all_idxs[k0]];
        int x0 = ptX[all_idxs[k0]];
        
        for(int k1=k0+1; k1 < all_idxs.size(); k1++){
            int y1 = ptY[all_idxs[k1]];
            int x1 = ptX[all_idxs[k1]];
           
            Aseg a_seg(y0, x0, y1, x1);
            m_all_seg[a_seg].fit_err = seg_info.GetData(k0, k1, 0);
            m_all_seg[a_seg].w[0]    = seg_info.GetData(k0, k1, 1);
            m_all_seg[a_seg].b[0]    = seg_info.GetData(k0, k1, 2);
            m_all_seg[a_seg].ch[0]   = dist_ch[0];
            
            m_all_seg[a_seg].w[1]    = seg_info.GetData(k0, k1, 3);
            m_all_seg[a_seg].b[1]    = seg_info.GetData(k0, k1, 4);
            m_all_seg[a_seg].ch[1]   = dist_ch[1];
        }
    }
    
    // record index information
    int y0 = ptY[all_idxs[0]];
    int x0 = ptX[all_idxs[0]];
    for(int k0=1; k0<all_idxs.size(); k0++){
        int y1 = ptY[all_idxs[k0]];
        int x1 = ptX[all_idxs[k0]];
        if(y1==y0){
            x1 = x1==m_wd-1? m_wd : x1;
            m_all_segIdx.ResetBulkData(all_idxs[k0-1], y0, 1, x0, x1-x0, e_seg_h*2, 1);
            m_all_segIdx.ResetBulkData(all_idxs[k0], y0, 1, x0, x1-x0, e_seg_h*2+1, 1);
        }
        else{
            y1 = y1==m_ht-1? m_ht : y1;
            m_all_segIdx.ResetBulkData(all_idxs[k0-1], y0, y1-y0, x0, 1, e_seg_v*2, 1);
            m_all_segIdx.ResetBulkData(all_idxs[k0], y0, y1-y0, x0, 1, e_seg_v*2+1, 1);
        }
        y0 = y1;  x0 = x1;
    }
}

void Segment_Stock::AssignDpSegments(CDataTempl<double> &seg_info, auto &semScore, vector<int> &dp_idxs, vector<int> &ptY, vector<int> &ptX){
    int seg_id = m_dp_seg.size()+1;
    
    // assign. 
    int y0 = ptY[dp_idxs[1]];
    int x0 = ptX[dp_idxs[1]];
    for(int ck=3; ck<dp_idxs.size(); ck+=2){
        int y1 = ptY[dp_idxs[ck]];
        int x1 = ptX[dp_idxs[ck]];
        int seg_len = dp_idxs[ck] - dp_idxs[ck-2]; 
        
        Mkey_2D key(dp_idxs[ck-3], dp_idxs[ck-1]);
        if(semScore[key][0] < 0.5){
            DpSeg dp_seg(seg_len, y0, x0, y1, x1);
            dp_seg.seg_info.fit_err = seg_info.GetData(dp_idxs[ck-3], dp_idxs[ck-1]);    
            dp_seg.seg_info.sem_score.assign(semScore[key].begin(), semScore[key].end());
            m_dp_seg[seg_id] = dp_seg;
      
            // record seg id to image.
            if(y1==y0){
                m_dp_segInfo.ResetBulkData(seg_id, y0, 1, x0, x1-x0, e_seg_h, 1);
#ifdef DEBUG_SEGMENT_STOCK
                m_dp_segInfo.ResetBulkData(ck-1, y0, 1, x0, x1-x0, e_seg_h, 1);
#endif
            }
            else{
                m_dp_segInfo.ResetBulkData(seg_id, y0, y1-y0, x0, 1, e_seg_v, 1);
#ifdef DEBUG_SEGMENT_STOCK
                m_dp_segInfo.ResetBulkData(ck-1, y0, y1-y0, x0, 1, e_seg_v, 1);
#endif
            }
            seg_id += 1;
        }

        // update for next loop.
        y0 = y1;   x0 = x1;
    }
}


double Segment_Stock::GetAllSegFitError(int y0, int x0, int y1, int x1){
    Aseg a_seg(y0, x0, y1, x1);
    return m_all_seg[a_seg].fit_err;
}

double Segment_Stock::GetAllSegFitErrorOnAny2Points(int y0, int x0, int y1, int x1){
    // find closest st, end.
    if(y0 == y1){
        x0 = m_all_segIdx.GetData(y0, x0, e_seg_h*2); 
        x1 = m_all_segIdx.GetData(y1, x1, e_seg_h*2+1); 
    }
    else{
        y0 = m_all_segIdx.GetData(y0, x0, e_seg_v*2); 
        y1 = m_all_segIdx.GetData(y1, x1, e_seg_v*2+1); 
    }
    
    return this->GetAllSegFitError(y0, x0, y1, x1);
}
SegFitRst &Segment_Stock::GetAllSegFitResultOnAny2Points(int y0, int x0, int y1, int x1){
    
    // find closest st, end.
    if(y0 == y1){
        x0 = m_all_segIdx.GetData(y0, x0, e_seg_h*2); 
        x1 = m_all_segIdx.GetData(y1, x1, e_seg_h*2+1); 
    }
    else{
        y0 = m_all_segIdx.GetData(y0, x0, e_seg_v*2); 
        y1 = m_all_segIdx.GetData(y1, x1, e_seg_v*2+1); 
    }
    
    // Get all_seg with updated (y0, x0, y1, x1). 
    Aseg a_seg(y0, x0, y1, x1);
    return m_all_seg[a_seg];
}

void Segment_Stock::GetDpSegmentById(DpSeg &dp_seg, int id){
    dp_seg = m_dp_seg[id];
}

void Segment_Stock::GetDpSegmentByCoord(DpSeg &dp_seg, int py, int px, SegType s_type){
    int id = m_dp_segInfo.GetData(py, px, s_type);
    this->GetDpSegmentById(dp_seg, id);
}


int Segment_Stock::GetDpSegmentSize(){
    return m_dp_seg.size();
}
#endif
