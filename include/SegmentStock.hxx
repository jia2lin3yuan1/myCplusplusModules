#ifndef _SEGMENT_STOCK_HXX
#define _SEGMENT_STOCK_HXX

#include "SegmentFitting.hxx"
#include "utils/read_write_img.hxx"

using namespace std;

/*
 * Class: SegmentStock.
 * There are three kind of segments in the stock:
 * 1. all_segment: possible segment locates between any two key points.
 * 2. dp_segment:  optimal segment comes from Dynamic Programming Partition in SegmentFitting.
 * 3. grow_segment: it could be segment between any two points in the same line (row or column), its cost is fitting cost of corresponding dp_segment + BIC_cost. 
 */
typedef struct All_Segment{
    UINT32 line;
    UINT32 st;
    UINT32 end;
    All_Segment(UINT32 a, UINT32 b, UINT32 c){
        line=a; st=b; end=c;
    } 
} Aseg; //[st, end)
struct AsegCmp{
    bool operator()(const Aseg& lhs, const Aseg& rhs)const{
        if(lhs.line != rhs.line)
            return lhs.line < rhs.line;
        else if(lhs.st != rhs.st)
            return lhs.st < rhs.st;
        else
            return lhs.end < rhs.end;
    }
};

typedef struct DP_Segment{
    UINT32 id;
    UINT32 line;
    UINT32 st;
    UINT32 end;
    float fit_err;
    DP_Segment(){
        id=-1; line=0; st=0; end=0; fit_err=0;
    }
    DP_Segment(UINT32 z, UINT32 a, UINT32 b, UINT32 c, float err){
        id=z; line=a; st=b; end=c; fit_err = err;
    } 
} DpSeg; // [st, end)

enum SegType {e_seg_h=0, e_seg_v,  e_seg_type_num};

class Segment_Stock{

protected:
    // Variables.
    map<Aseg, float, AsegCmp> m_all_seg_h;
    map<Aseg, float, AsegCmp> m_all_seg_v;
    vector<DpSeg>     m_dp_seg_h;
    vector<DpSeg>     m_dp_seg_v;

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

    void AssignAllSegments(CDataTempl<float> &fit_err, vector<UINT32> &all_idxs, UINT32 line, bool is_row);
    void AssignDpSegments(CDataTempl<float> &fit_err, vector<UINT32> &dp_idxs, UINT32 line, bool is_row);
   

    float GetAllSegCost(UINT32 line, UINT32 st, UINT32 end, bool is_row);
    void GetDpSegment(DpSeg &dp_seg, UINT32 id, bool is_row);
    void GetDpSegment(DpSeg &dp_seg, UINT32 py, UINT32 px, bool is_row);
    UINT32 GetDpSegmentSize(bool is_row);
    
};


void Segment_Stock::AssignAllSegments(CDataTempl<float> &fit_err, vector<UINT32> &all_idxs, UINT32 line, bool is_row){
    UINT32 line_len = is_row? m_wd : m_ht;
    // assign.
    for(UINT32 k0=0; k0<all_idxs.size()-1; k0++){
        UINT32 st  = all_idxs[k0];
        for(UINT32 k1=k0+1; k1<all_idxs.size(); k1++){
            UINT32 end = all_idxs[k1]==line_len-1? line_len : all_idxs[k1];
            Aseg a_seg(line, st, end);
            if(is_row)
                m_all_seg_h[a_seg] = fit_err.GetData(k0, k1);
            else
                m_all_seg_v[a_seg] = fit_err.GetData(k0, k1);
        }
    }
}

void Segment_Stock::AssignDpSegments(CDataTempl<float> &fit_err, vector<UINT32> &dp_idxs, UINT32 line, bool is_row){
    UINT32 seg_id   = is_row? m_dp_seg_h.size() : m_dp_seg_v.size();
    UINT32 line_len = is_row? m_wd : m_ht;
    // assign. 
    UINT32 pt_st = dp_idxs[1], pt_end = 0;
    for(UINT32 k=3; k<dp_idxs.size(); k+=2){
        pt_end = dp_idxs[k]==line_len-1? line_len : dp_idxs[k];
        DpSeg dp_seg(seg_id, line, pt_st, pt_end, fit_err.GetData(dp_idxs[k-3], dp_idxs[k-1]));
        
        if(is_row){
            m_dp_seg_h.push_back(dp_seg);
            m_dp_segInfo.ResetBulkData(seg_id, line, 1, pt_st, pt_end-pt_st, e_seg_h, 1);
#ifdef DEBUG_SEGMENT_STOCK
            m_dp_segInfo.ResetBulkData(pt_end, line, 1, pt_st, pt_end-pt_st, e_seg_h, 1);
#endif
        }
        else{
            m_dp_seg_v.push_back(dp_seg);
            m_dp_segInfo.ResetBulkData(seg_id, pt_st, pt_end-pt_st, line, 1, e_seg_v, 1);
#ifdef DEBUG_SEGMENT_STOCK
            m_dp_segInfo.ResetBulkData(pt_end, pt_st, pt_end-pt_st, line, 1, e_seg_v, 1);
#endif
        }

        // update for next loop.
        seg_id += 1;
        pt_st   = pt_end;
    }
}

float Segment_Stock::GetAllSegCost(UINT32 line, UINT32 st, UINT32 end, bool is_row){
    Aseg a_seg(line, st, end);
    if(is_row)
        return m_all_seg_h[a_seg];
    else
        return m_all_seg_v[a_seg];
}

void Segment_Stock::GetDpSegment(DpSeg &dp_seg, UINT32 id, bool is_row){
    if(is_row)
        dp_seg = m_dp_seg_h[id];
    else
        dp_seg = m_dp_seg_v[id];
}

void Segment_Stock::GetDpSegment(DpSeg &dp_seg, UINT32 py, UINT32 px, bool is_row){
    UINT32 id = is_row? m_dp_segInfo.GetData(py, px, e_seg_h) : m_dp_segInfo.GetData(py, px, e_seg_v);
    this->GetDpSegment(dp_seg, id, is_row);
}


UINT32 Segment_Stock::GetDpSegmentSize(bool is_row){
    if (is_row)
        return m_dp_seg_h.size();
    else
        return m_dp_seg_v.size();
}
#endif
