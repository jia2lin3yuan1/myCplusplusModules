#ifndef SEGMENT_GROW_HXX
#define SEGMENT_GROW_HXX


#include "DataTemplate.hxx"
#include "SegmentFitting.hxx"

using namespace std;

double glb_BIC_LUT[] = { // size=100, 1/log(2), 1/log(2.5), ..., 1/log(51.5)
    1.44269504,  1.09135667,  0.91023923,  0.7982356 ,  0.72134752,
    0.6648594 ,  0.62133493,  0.58659693,  0.55811063,  0.53424449,
    0.51389834,  0.4963018 ,  0.48089835,  0.46727527,  0.45511961,
    0.44418942,  0.43429448,  0.42528303,  0.41703239,  0.40944222,
    0.4024296 ,  0.39592535,  0.38987125,  0.38421791,  0.37892318,
    0.37395079,  0.36926937,  0.36485165,  0.36067376,  0.35671475,
    0.35295612,  0.34938149,  0.34597626,  0.34272741,  0.33962327,
    0.33665336,  0.3338082 ,  0.33107925,  0.32845874,  0.32593962,
    0.32351545,  0.32118037,  0.31892899,  0.31675637,  0.31465798,
    0.31262963,  0.31066747,  0.30876792,  0.30692768,  0.30514368,
    0.30341308,  0.30173322,  0.30010163,  0.29851601,  0.2969742 ,
    0.2954742 ,  0.2940141 ,  0.29259215,  0.29120668,  0.28985612,
    0.28853901,  0.28725396,  0.28599967,  0.2847749 ,  0.28357849,
    0.28240934,  0.28126641,  0.28014872,  0.27905531,  0.27798532,
    0.27693789,  0.27591223,  0.27490758,  0.2739232 ,  0.27295842,
    0.27201257,  0.27108503,  0.2701752 ,  0.26928251,  0.26840641,
    0.26754639,  0.26670194,  0.26587259,  0.26505788,  0.26425737,
    0.26347065,  0.26269731,  0.26193697,  0.26118926,  0.26045381,
    0.2597303 ,  0.25901839,  0.25831777,  0.25762812,  0.25694917,
    0.25628063,  0.25562222,  0.25497369,  0.25433478,  0.25370525
};    
    

// Claim of Non-class functions.
static string ConstructKey(UINT32 v0, UINT32 v1, UINT32 v2, UINT32 len=5){
    string keyS = _Int2String(v0, len)+"-"+_Int2String(v1, len)+"-"+_Int2String(v2, len);
    return keyS;
}

static void ParseKey(string keyS, UINT32 &v0, UINT32 &v1, UINT32 &v2, UINT32 len=5){
    v0 = _String2Int(keyS.substr(0, len));
    v1 = _String2Int(keyS.substr(len+1, len));
    v2 = _String2Int(keyS.substr(2*len+2, len));
}

/*
class Segment_Region{
    

};
*/

typedef struct{
    UINT32 seg_id; 
    UINT32 st, end;
    double fit_err;
    double seg_dist;      
}Segment_info;

template<typename COST>
class PQ_cost{ // support maximum 3 keys.
protected:
    struct compare{
        bool operator()(const pair<string, COST> &lhs, const pair<string, COST> &rhs){
            return lhs.second > rhs.second;
        }
    };
    
    UINT32  m_key_len;
    priority_queue<pair<string, COST>, vector<pair<string, COST> >, compare> m_queue;
    map<string, UINT8> m_keys_map; 
public:
    PQ_cost(){
        m_key_len = 5;
    }
    void push(COST cost, UINT32 k0, UINT32 k1=0, UINT32 k2=0){
        string keyS = ConstructKey(k0, k1, k2, m_key_len);
        m_queue.push(make_pair(keyS, cost));
        m_keys_map[keyS] = 0;
    }
    UINT8 top(COST &cost, UINT32 &k0, UINT32 &k1, UINT32 &k2){
        while(!m_queue.empty() && m_keys_map.find(m_queue.top().first) == m_keys_map.end()){
            m_queue.pop();
        }
        
        if(m_queue.empty())
            return 0;
        else{
            cost  = m_queue.top().second;
            ParseKey(m_queue.top().first, k0, k1, k2, m_key_len);
            return 1;
        }
    }
    UINT8 pop(COST &cost, UINT32 &k0, UINT32 &k1, UINT32 &k2){
        UINT8 rst = this->top(cost, k0, k1, k2);
        if(rst > 0){
            m_keys_map.erase(m_queue.top().first);
            m_queue.pop();
        }

        return rst;
    }
    
    bool empty() const{
        return m_queue.empty();
    }
    auto size() const{
        return m_keys_map.size();
    }

    void deleteKey(UINT32 k0, UINT32 k1, UINT32 k2){
        string keyS = ConstructKey(k0, k1, k2, m_key_len);
        m_keys_map.erase(keyS);
    }
};


 
enum SegDP_type {es_stH=0, es_endH, es_errH, es_stV, es_endV, es_errV, es_chNum};
enum Mask_type {ms_BG=0, ms_FG, ms_POS_FG};
class Segment_Grow{
protected:
    CDataTempl<UINT32>   m_maskI; // segment grow mask. 0-unprocessed data, 1-background. >1 - grown region label.
    map<string, double> m_seg_hmap; // all possible H/V segments. key is 'line-st-end'. value is the fitting error.
    map<string, double> m_seg_vmap;
    PQ_cost<double> m_segSeedsQ; // sorted segment seeds based on the cost. The string is "py-stH-endH".
    
    UINT32 m_ht, m_wd;
    CDataTempl<UINT32>  m_seg_dpI; // channel = 6, refer to SegDP_type in line 53 from DP in SegmentFitting.
    CDataTempl<UINT32>  m_seg_updI; // initial equals to m_seg_dpI, then would be cutted to small segments in processing.
    
    // tmp variables for each shrink process in growing.
    vector<pair<UINT32, UINT32> > m_borderH; // used to record the border pixel for mask 1 in each line
    vector<pair<UINT32, UINT32> > m_borderV; // used to record the border pixel for mask 1 in each column

    // parameters in computation.
    double m_seed_bic_alpha, m_seed_bic_scale;
    double m_rm_bic_alpha, m_rm_fit_cost_thr, m_rm_fit_cost_penalty, m_rm_cost_thr;
    UINT32 m_prop_size_thr, m_seed_size_thr;
    UINT32 m_rm_bic_addi_len_oft;

public:
    Segment_Grow(UINT32 ht, UINT32 wd){
        m_ht = ht;  m_wd = wd;
        m_seg_dpI.Init(m_ht, m_wd, es_chNum);
        m_seg_updI.Init(m_ht, m_wd, es_chNum);
        m_maskI.Init(m_ht, m_wd);

        SetupConfig();
    }
    CDataTempl<UINT32>& GetFinalResult(){
        return m_maskI;
    }
    void SetupConfig();
    
    void AssignAllSegment( CDataTempl<double> &fit_err, vector<UINT32> &seg_iniIdx, UINT32 line, bool isRow);
    void AssignDPSegment_H(CDataTempl<UINT32> &sem_bgI, CDataTempl<double> &fit_err, vector<UINT32> &seg_dpIdx, UINT32 py);
    void AssignDPSegment_V(CDataTempl<UINT32> &sem_bgI, CDataTempl<double> &fit_err, vector<UINT32> &seg_dpIdx, UINT32 px);

    // Generate a label image with segment growing.
    void ImagePartition(CDataTempl<UINT32> &sem_bgI);
    void GrowingFromASegment(UINT32 py, UINT32 st, UINT32 end, UINT32 &propId);
    void GrowingExtend(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor);
    void GrowingShrink(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor);

    double ComputeOneNodeCost(CDataTempl<UINT32> &mask, UINT32 py, UINT32 px, bool ext_hor);

};

// compute the cost of keep (py, px)
double Segment_Grow::ComputeOneNodeCost(CDataTempl<UINT32> &mask, UINT32 py, UINT32 px, bool ext_hor){
    auto relocate_start_end = [&](UINT32 &seg_st, UINT32 &seg_end, UINT32 &mask_st, UINT32 &mask_end, bool isRow){
        seg_st   = m_seg_updI.GetData(py, px, (isRow? es_stH:es_stV));
        seg_end  = m_seg_updI.GetData(py, px, (isRow? es_endH:es_endV));
        mask_end = mask.FindMaskBorderPoint(py, px, seg_st, seg_end, (isRow? 1 : m_wd));
        mask_st  = isRow? px : py;
        if(mask_end >= mask_st){
            seg_end = seg_st;
        }

        seg_st = mask_end;
    };

    auto ComputeBICcost = [&](UINT32 x0, UINT32 x1, UINT32 bias){
        return glb_BIC_LUT[_CLIP(UINT32(abs(x1-x0)+bias+m_rm_bic_addi_len_oft), 0, 99)];
    };
    auto ComputeFitCost = [&](UINT32 py, UINT32 px){
        // H direction.
        UINT32 st_x  = m_seg_dpI.GetData(py, px, es_stH);
        UINT32 end_x = m_seg_dpI.GetData(py, px, es_endH);
        UINT32 bd_x0 = m_borderH[py].first;
        UINT32 bd_x1 = m_borderH[py].second;
        if(bd_x0 < bd_x1){
            st_x  = min(st_x, m_seg_dpI.GetData(py, bd_x0, es_stH));
            end_x = max(end_x, m_seg_dpI.GetData(py, bd_x1, es_endH));
        }
        double fit_cost_h = m_seg_hmap[ConstructKey(py, st_x, end_x)];
        fit_cost_h = fit_cost_h + m_rm_fit_cost_penalty*(fit_cost_h>m_rm_fit_cost_thr);
        // V direction.
        UINT32 st_y  = m_seg_dpI.GetData(py, px, es_stV);
        UINT32 end_y = m_seg_dpI.GetData(py, px, es_endV);
        UINT32 bd_y0 = m_borderV[px].first;
        UINT32 bd_y1 = m_borderV[px].second;
        if(bd_y0 < bd_y1){
            st_y  = min(st_y, m_seg_dpI.GetData(bd_y0, px, es_stV));
            end_y = max(end_y, m_seg_dpI.GetData(bd_y1, px, es_endV));
        }
        double fit_cost_v = m_seg_vmap[ConstructKey(px, st_y, end_y)];
        fit_cost_v = fit_cost_v + m_rm_fit_cost_penalty*(fit_cost_v>m_rm_fit_cost_thr);

        return fit_cost_h + fit_cost_v;
    };

    //find st/end for (py, px), and mask_st, mask_end inside mask from (py, px).
    UINT32 seg_st_A, seg_end_A, mask_end_A, mask_st_A;
    UINT32 seg_st_B, seg_end_B, mask_end_B, mask_st_B;

    // st, end, cut position.
    if(ext_hor){
        relocate_start_end(seg_st_A, seg_end_A, mask_st_A, mask_end_A, 1);
        relocate_start_end(seg_st_B, seg_end_B, mask_st_B, mask_end_B, m_wd);
    }
    else{
        relocate_start_end(seg_st_B, seg_end_B, mask_st_B, mask_end_B, 1);
        relocate_start_end(seg_st_A, seg_end_A, mask_st_A, mask_end_A, m_wd);
    }

    // compute BIC cost and FIT cost
    UINT32 len_A = abs(seg_end_A - seg_st_A) + 1;
    double bic_A = ComputeBICcost(mask_st_A, seg_st_A, 1);
    bic_A += ComputeBICcost(mask_st_A, seg_end_A, 1+0.1*len_A);
    bic_A -= ComputeBICcost(mask_st_A, seg_st_A, 0);
    bic_A -= ComputeBICcost(mask_st_A, seg_end_A, 2+0.1*len_A);

    UINT32 len_B = abs(seg_end_B - seg_st_B) + 1;
    double bic_B = ComputeBICcost(mask_st_B, seg_st_B, 1);
    bic_B += ComputeBICcost(mask_st_B, seg_end_B, 1+0.2*len_B);
    bic_B -= ComputeBICcost(mask_st_B, seg_st_B, 0);
    bic_B -= ComputeBICcost(mask_st_B, seg_end_B, 2+0.2*len_B);

    double fit_A = ComputeFitCost(py, px);

    return -(fit_A + m_rm_bic_alpha*(bic_A+bic_B));
}

void Segment_Grow::GrowingShrink(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor){
    auto ComputeCost = [&](UINT32 py, UINT32 px, auto &cost_heap){
        // if it's not possible FG, or it's not a corner point, return directly.
        if(mask.GetData(py, px) != ms_POS_FG)
            return;
        else if((px > 0 && mask.GetData(py, px-1)>0 && px < m_wd-1 && mask.GetData(py, px+1))>0)
            return;
        else if(py > 0 && mask.GetData(py-1, px)>0 && py < m_ht-1 && mask.GetData(py+1, px)>0)
            return;
        else if (mask.GetData(py, px) == 2){
            double rm_cost = ComputeOneNodeCost(mask, py, px, ext_hor);
            cost_heap.push(rm_cost, py, px, px);
        }
    };

    auto UpdateSegUpdI = [&](UINT32 py, UINT32 px, UINT32 st, UINT32 mid, UINT32 end, double cost0, double cost1, bool is_hor){
        if(is_hor){
            m_seg_updI.ResetBulkData(cost0, py, 1, st, mid-st, es_errH, 1);
            m_seg_updI.ResetBulkData(cost1, py, 1, mid, end-mid+1, es_errH, 1);
            m_seg_updI.ResetBulkData(mid-1, py, 1, st, mid-st, es_endH, 1);
            m_seg_updI.ResetBulkData(mid, py, 1, mid, end-mid+1, es_stH, 1);
        }
        else{
            m_seg_updI.ResetBulkData(cost0, st,mid-st, px, 1, es_errV, 1);
            m_seg_updI.ResetBulkData(cost1, mid, end-mid+1, px, 1, es_errV, 1);
            m_seg_updI.ResetBulkData(mid-1, st, mid-st, px, 1, es_endV, 1);
            m_seg_updI.ResetBulkData(mid, mid, end-mid+1, px, 1, es_stV, 1);
        }
    };

    auto UpdateSegments = [&](UINT32 py, UINT32 px, bool is_hor){
        // update recording of segments in m_seg_updI after being cutted. 
        UINT32 st  = m_seg_updI.GetData(py, px, (is_hor? es_stH : es_stV));
        UINT32 end = m_seg_updI.GetData(py, px, (is_hor? es_endH: es_endV));
        if(is_hor)
            m_segSeedsQ.deleteKey(py, st, end);
        
        // compute new st/end and error for each sub-segment.
        double fit_cost  = m_seg_dpI.GetData(py, px, (is_hor? es_errH : es_errV));
        bool mask_on_end = is_hor? mask.GetData(py, px+1) != ms_BG : mask.GetData(py+1, px)!=ms_BG;
        if(mask_on_end){
            UINT32 mid = is_hor? px : py;
            if(mid == st){ // if no cut.
                m_seg_updI.ResetBulkData(1e3, py, 1, st, end-st+1, es_errH, 1);
                return;
            }

            UINT32 BIC_cost_c  = _CLIP(int((mid-st)*m_seed_bic_scale), 0, 99); 
            double st_cost  = fit_cost + m_seed_bic_alpha*glb_BIC_LUT[BIC_cost_c];
            UpdateSegUpdI(py, px, st, mid, end, st_cost, 1e3, is_hor);
            if(is_hor && mid > st + m_seed_size_thr)
                m_segSeedsQ.push(st_cost, py, st, mid-1);
        }
        else{
            UINT32 mid = is_hor? px+1 : py+1;
            if(mid > end){// if no cut.
                m_seg_updI.ResetBulkData(1e3, py, 1, st, end-st+1, es_errH, 1);
                return;
            }
            
            UINT32 BIC_cost_c =_CLIP(int((mid-st)*m_seed_bic_scale), 0, 99); 
            double end_cost  = fit_cost + m_seed_bic_alpha*glb_BIC_LUT[BIC_cost_c];
            UpdateSegUpdI(py, px, st, mid, end, 1e3, end_cost, is_hor);
            if(is_hor && end > mid - 1 + m_seed_size_thr)
                m_segSeedsQ.push(end_cost, py, mid, end);
        }

    };

    // starting from corner points on the boundary.
    PQ_cost<double> bdPix_pq;
    for(UINT32 k=0; k<bdPair.size(); k++){
        ComputeCost(bdPair[k].first, bdPair[k].second, bdPix_pq);
    }

    while(!bdPix_pq.empty()){
        UINT32 py, px, tmp;
        double rm_cost; 
        bdPix_pq.pop(rm_cost, py, px, tmp);
        if(rm_cost < m_rm_cost_thr){ // remove the pixel
            mask.SetData(ms_BG, py, px);
            
            if (py>0 && mask.GetData(py-1, px)==2)
                ComputeCost(py-1, px, bdPix_pq);
            if (py<m_ht-1 && mask.GetData(py+1, px)==2)
                ComputeCost(py+1, px, bdPix_pq);
            if (px>0 && mask.GetData(py, px-1)==2)
                ComputeCost(py, px-1, bdPix_pq);
            if (px<m_wd-1 && mask.GetData(py, px+1)==2)
                ComputeCost(py, px+1, bdPix_pq);
        }
        else{ // if the pixel is retained.
            UpdateSegments(py, px, true);
            UpdateSegments(py, px, false);
        }
    }
}


void Segment_Grow::GrowingExtend(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor){
    auto GetMaskData = [&](UINT32 line, UINT32 sk){ 
        return ext_hor? mask.GetData(line, sk) : mask.GetData(sk, line);
    };
    auto SetMaskData = [&](UINT32 val, UINT32 line, UINT32 sk){ 
        return ext_hor? mask.SetData(val, line, sk) : mask.SetData(val, sk, line);
    };
    auto GetStEnd =[&](UINT32 py, UINT32 px, UINT32 &st, UINT32 &end){
        if(ext_hor){
            st  = m_seg_updI.GetData(py, px, es_stH);
            end = m_seg_updI.GetData(py, px, es_endH);
        }
        else{
            st  = m_seg_updI.GetData(py, px, es_stV);
            end = m_seg_updI.GetData(py, px, es_endV);
        }
    };
    auto Insert2bdPair = [&](UINT32 line, UINT32 sk){
        if(ext_hor){
            bdPair.push_back(make_pair(line, sk));
        }
        else{
            bdPair.push_back(make_pair(sk, line));
        }
    };
    
    // starting process.
    vector<pair<UINT32, UINT32> > pre_bds;
    mask.FindBoundaryOnMask(pre_bds, 1);
    
    // extend from boundary pixels.
    m_borderH.clear();
    m_borderH.resize(m_ht, make_pair(m_wd, 0));
    m_borderV.clear();
    m_borderV.resize(m_wd, make_pair(m_ht, 0));
    for(int k=0; k < pre_bds.size(); k++){
        UINT32 py = pre_bds[k].first;
        UINT32 px = pre_bds[k].second;
        if(m_maskI.GetData(py, px) > 0)
            continue;
        
        // update (min,max) border for each row, column
        m_borderH[py].first  = px < m_borderH[py].first?  px : m_borderH[py].first;
        m_borderH[py].second = px > m_borderH[py].second? px : m_borderH[py].second;
        m_borderV[px].first  = py < m_borderV[px].first?  py : m_borderV[px].first;
        m_borderV[px].second = py > m_borderV[px].second? py : m_borderV[px].second;
        
        // update curMask. 
        UINT32 st = 0, end = 0;
        GetStEnd(py, px, st, end);
        UINT32 line = ext_hor? py : px;
        for(int sk = st; sk <= end; sk++){
            if(GetMaskData(line, sk) == ms_BG)
                SetMaskData(ms_POS_FG, line, sk);
        }

        // set edge pixel to bdPair, used as seed for shrink processing.
        Insert2bdPair(line, st);
        Insert2bdPair(line, end);
    }
}

void Segment_Grow::GrowingFromASegment(UINT32 py, UINT32 st, UINT32 end, UINT32 &propId){
        
    CDataTempl<UINT32> curMask(m_ht, m_wd);
    curMask.ResetBulkData(ms_FG, py, 1, st, end-st+1);

    UINT32 grow_tot = 0, grow_1step = 30; 
    bool ext_hor = false;
    while(grow_1step>0){
        vector<pair<UINT32, UINT32> > bdPair;
        GrowingExtend(curMask, bdPair, ext_hor);
        GrowingShrink(curMask, bdPair, ext_hor);
        
        grow_1step = curMask.ReplaceByValue(ms_POS_FG, ms_FG);
        grow_tot  += grow_1step;
        ext_hor    = !ext_hor;
    }

    if(grow_tot > m_prop_size_thr){
        curMask.ModifyMaskOnNonZeros(m_maskI, propId);
        propId ++;
    }
}


void Segment_Grow::ImagePartition(CDataTempl<UINT32> &sem_bgI){
    m_seg_updI.Copy(m_seg_dpI);
    m_maskI.Copy(sem_bgI);

    UINT32 propId = 2;
    while(true){
        // pop out seed, and check if it valid.
        double cost;
        UINT32 py, st, end;
        if(m_segSeedsQ.pop(cost, py, st, end) == 0){
            break; // there is no available seeds.
        }
        
        GrowingFromASegment(py, st, end, propId);
    }
}



void Segment_Grow::AssignAllSegment(CDataTempl<double> &fit_err, vector<UINT32> &seg_iniIdx, UINT32 line, bool isRow=true){
    map<string, double> &ref_map = isRow? m_seg_hmap : m_seg_vmap;
    // adding to the whole map
    for(UINT32 k0=0; k0<seg_iniIdx.size()-1; k0++){
        for(UINT32 k1=k0+1; k1<seg_iniIdx.size(); k1++){
            UINT32 st = seg_iniIdx[k0], end=seg_iniIdx[k1];
            string mkey = ConstructKey(line, st, end);
            ref_map[mkey] = fit_err.GetData(st, end);
        }
    }
}

void Segment_Grow::AssignDPSegment_H(CDataTempl<UINT32> &sem_bgI, CDataTempl<double> &fit_err, vector<UINT32> &seg_dpIdx, UINT32 py){
    // adding dp result to seed.
    for(UINT32 k=0; k<seg_dpIdx.size()-1; k++){
        UINT32 st = seg_dpIdx[k], end=seg_dpIdx[k+1];
        double seg_cost = 1e3;
        if (sem_bgI.GetData(py, st+1) == 0){
            int lut_idx = int(_CLIP(int((end-st+1)*m_seed_bic_scale), 0, 99));
            seg_cost = fit_err.GetData(st,end) + m_seed_bic_alpha*glb_BIC_LUT[lut_idx];
            m_segSeedsQ.push(seg_cost, py, st, end);
        }

        m_seg_dpI.ResetBulkData(st, py, 1, st, end-st+1, es_stH, 1);
        m_seg_dpI.ResetBulkData(end, py, 1, st, end-st+1, es_endH, 1);
        m_seg_dpI.ResetBulkData(seg_cost, py, 1, st, end-st+1, es_errH, 1);
    }
}
void Segment_Grow::AssignDPSegment_V(CDataTempl<UINT32> &sem_bgI, CDataTempl<double> &fit_err, vector<UINT32> &seg_dpIdx, UINT32 px){
    // adding dp result to seed.
    for(UINT32 k=0; k<seg_dpIdx.size()-1; k++){
        UINT32 st = seg_dpIdx[k], end=seg_dpIdx[k+1];
        double seg_cost = 1e3;
        if (sem_bgI.GetData(st+1, px) == 0){
            int lut_idx = int(_CLIP(int((end-st+1)*m_seed_bic_scale), 0, 99));
            seg_cost = fit_err.GetData(st,end) + m_seed_bic_alpha*glb_BIC_LUT[lut_idx];
        }
        
        m_seg_dpI.ResetBulkData(st, st, end-st+1, px, 1, es_stV, 1);
        m_seg_dpI.ResetBulkData(end, st, end-st+1, px, 1, es_endV, 1);
        m_seg_dpI.ResetBulkData(seg_cost, st, end-st+1, px, 1, es_errV, 1);
    }
}

void Segment_Grow::SetupConfig(){

    m_seed_bic_alpha = 1e-1;
    m_seed_bic_scale = 2e-1;

    m_rm_bic_alpha = 1;
    m_rm_fit_cost_thr = 2e-2;
    m_rm_fit_cost_penalty = 1e3;
    m_rm_cost_thr = 0;

    m_prop_size_thr = 40;
    m_seed_size_thr = 5;

    m_rm_bic_addi_len_oft=1;
}


#endif
