#ifndef SEGMENT_GROW_HXX
#define SEGMENT_GROW_HXX

#include "utils/read_write_img.hxx"
#include "SegmentFitting.hxx"
#include "SegmentStock.hxx"

using namespace std;
 
typedef struct Grow_Segment{
    UINT32 id;
    UINT32 line;
    UINT32 st;
    UINT32 end;
    float fit_err;
    float bic_cost;
    bool   valid;
    Grow_Segment(){
        id=-1; line=0; st=0; end=0; fit_err=0; bic_cost=0; valid=false;
    }
    Grow_Segment(UINT32 z, UINT32 a, UINT32 b, UINT32 c, float err, float cost, bool flag=true){
        id=z; line=a; st=b; end=c; fit_err=err; bic_cost=cost; valid=flag;
    }
    Grow_Segment(DpSeg &seg, float cost, bool flag=true){
        id = seg.id; line=seg.line; st=seg.st; end=seg.end; fit_err=seg.fit_err; 
        bic_cost=cost; valid=flag;
    }
} GrowSeg; // [st, end)

enum Mask_type {ms_BG=0, ms_FG, ms_POS_FG};

/*
 * Class: Segment_Grow.
 *        Based on segment, do region growing.
 *
 */

class Segment_Grow{
protected:
    // variables exist in the whole life.
    UINT32 m_ht, m_wd;
    CDataTempl<float>    m_out_debugI;
    CDataTempl<UINT32>   m_out_maskI; // segment grow mask. 0-unprocessed data, 1-background. >1 - grown region label.
    Segment_Stock *      m_pSegStock; // segment informations 

    CDataTempl<UINT32>   m_grow_segInfo;
    vector<GrowSeg>      m_grow_seg_h;
    vector<GrowSeg>      m_grow_seg_v;
    priority_queue<Seed, vector<Seed>, SeedCmp> m_grow_seg_seeds;
    
    // tmp variables for each shrink process in growing.
    vector<pair<UINT32, UINT32> > m_borderH; // used to record the border pixel of current mask in each line
    vector<pair<UINT32, UINT32> > m_borderV; // ... in each column

    // parameters.
    const GlbParam *m_pParam;

public:
    Segment_Grow(UINT32 ht, UINT32 wd, const CDataTempl<UINT32> &sem_bgI, Segment_Stock * pSegStock,  const GlbParam *pParam){
        m_ht = ht;  m_wd = wd;
        m_out_debugI.Init(m_ht, m_wd);
        m_out_maskI.Init(m_ht, m_wd);
        
        m_pSegStock = pSegStock;
        m_pParam    = pParam;
        
        m_grow_segInfo.Init(m_ht, m_wd, e_seg_type_num);
        InitialSetGrowSegments(sem_bgI, true);
        InitialSetGrowSegments(sem_bgI, false);
    }
    CDataTempl<UINT32>& GetFinalResult(){
        return m_out_maskI;
    }
    CDataTempl<float>& GetDebugInformation();
    
    
    // operations on "grow segments"
    float ComputeGrowSegmentBICcost(UINT32 len);
    void InitialSetGrowSegments(const CDataTempl<UINT32> &sem_bgI, bool is_row);
    bool GrowSegmentIsValid(UINT32 id, bool is_row);
    void GetGrowSegment(GrowSeg &gw_seg, UINT32 id, bool is_row);
    void GetGrowSegment(GrowSeg &gw_seg, UINT32 py, UINT32 px, bool is_row);
    void DisableGrowSegment(UINT32 id, bool is_row);
    void AddGrowSegment(UINT32 line, UINT32 st, UINT32 end, bool is_row);

    // Generate a label image with segment growing.
    void ImagePartition(CDataTempl<UINT32> &sem_bgI);
    void GrowingFromASegment(UINT32 grow_seed_id, UINT32 &propId);
    void GrowingExtend(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor);
    void GrowingShrink(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor);
    float ComputeOneNodeCost(CDataTempl<UINT32> &mask, UINT32 py, UINT32 px, bool ext_hor);

    void ResetBorderHV();
};

CDataTempl<float>& Segment_Grow::GetDebugInformation(){
    UINT32 num_Seg = m_grow_seg_h.size();
    for(UINT32 k =0; k < num_Seg; k++){
        GrowSeg gw_seg = m_grow_seg_h[k];
        m_out_debugI.ResetBulkData(gw_seg.fit_err+gw_seg.bic_cost, gw_seg.line, 1, gw_seg.st, gw_seg.end-gw_seg.st); 
    }
    return m_out_debugI;
}

// function for managing growing segments.
float Segment_Grow::ComputeGrowSegmentBICcost(UINT32 len){
    //int   lut_idx  = int(len*m_pParam->segGrow_seed_bic_scale);
    //float bic_cost = m_pParam->segGrow_seed_bic_alpha * InvLog_LUT(lut_idx);
    float   lut_idx = float(len)*m_pParam->segGrow_seed_bic_scale;
    float bic_cost  = m_pParam->segGrow_seed_bic_alpha * (1/log(lut_idx+2));
    return bic_cost;
}
void Segment_Grow::InitialSetGrowSegments(const CDataTempl<UINT32> &sem_bgI, bool is_row){
    UINT32 num_dpSeg = m_pSegStock->GetDpSegmentSize(is_row);
    for(UINT32 k =0; k < num_dpSeg; k++){
        DpSeg dp_seg;
        m_pSegStock->GetDpSegment(dp_seg, k, is_row);

        float bic_cost = ComputeGrowSegmentBICcost(dp_seg.end-dp_seg.st);
        GrowSeg gw_seg(dp_seg, bic_cost, true);
        
        if(is_row){
            // set invalid according to sem_bgI
            if(sem_bgI.GetData(dp_seg.line, dp_seg.st+1)==1){
                gw_seg.valid=false;
                gw_seg.fit_err = 1e3;
            }
            m_grow_seg_h.push_back(gw_seg);
            m_grow_segInfo.ResetBulkData(k, dp_seg.line, 1, dp_seg.st, dp_seg.end-dp_seg.st, e_seg_h, 1);
     
            // if the segment is valid, add into segment seed.
            if(gw_seg.valid){
                Seed seg_seed(k, k, gw_seg.bic_cost+gw_seg.fit_err);
                m_grow_seg_seeds.push(seg_seed);
            }
        }
        else{
            m_grow_seg_v.push_back(gw_seg);
            m_grow_segInfo.ResetBulkData(k, dp_seg.st, dp_seg.end-dp_seg.st, dp_seg.line, 1, e_seg_v, 1);
        }
    }
}
bool Segment_Grow::GrowSegmentIsValid(UINT32 id, bool is_row){
    return (is_row? m_grow_seg_h[id].valid : m_grow_seg_v[id].valid);
}

void Segment_Grow::GetGrowSegment(GrowSeg &gw_seg, UINT32 id, bool is_row){
    if(is_row)
        gw_seg = m_grow_seg_h[id];
    else
        gw_seg = m_grow_seg_v[id];
}

void Segment_Grow::GetGrowSegment(GrowSeg &gw_seg, UINT32 py, UINT32 px, bool is_row){
    UINT32 id = is_row? m_grow_segInfo.GetData(py, px, e_seg_h) : m_grow_segInfo.GetData(py, px, e_seg_v);
    this->GetGrowSegment(gw_seg, id, is_row);
}

void Segment_Grow::DisableGrowSegment(UINT32 id, bool is_row){
    if(is_row)
        m_grow_seg_h[id].valid = false;
    else
        m_grow_seg_v[id].valid = false;
}

void Segment_Grow::AddGrowSegment(UINT32 line, UINT32 st, UINT32 end, bool is_row){
    float bic_cost = ComputeGrowSegmentBICcost(end-st);
    if(is_row){
        UINT32 seg_id  = m_grow_seg_h.size();
        DpSeg dp_seg;
        m_pSegStock->GetDpSegment(dp_seg, line, st, true);
        GrowSeg gw_seg(seg_id, line, st, end, dp_seg.fit_err, bic_cost, true);
        m_grow_seg_h.push_back(gw_seg);
        m_grow_segInfo.ResetBulkData(seg_id, line, 1, st, end-st, e_seg_h, 1);
        
        // add of segment seed.
        Seed seg_seed(seg_id, seg_id, gw_seg.bic_cost+gw_seg.fit_err);
        m_grow_seg_seeds.push(seg_seed);
    }
    else{
        UINT32 seg_id  = m_grow_seg_v.size();
        DpSeg dp_seg;
        m_pSegStock->GetDpSegment(dp_seg, st, line, false);
        GrowSeg gw_seg(line, st, end, dp_seg.fit_err, bic_cost, true);
        m_grow_seg_v.push_back(gw_seg);
        m_grow_segInfo.ResetBulkData(seg_id, st, end-st, line, 1, e_seg_v, 1);
    }
}

//  Functions for growing based on segments.

void Segment_Grow::ResetBorderHV(){
    m_borderH.clear();
    m_borderH.resize(m_ht, make_pair(m_wd, 0));
    m_borderV.clear();
    m_borderV.resize(m_wd, make_pair(m_ht, 0));
}

// compute the cost of keep (py, px)
float Segment_Grow::ComputeOneNodeCost(CDataTempl<UINT32> &mask, UINT32 py, UINT32 px, bool ext_hor){
    auto relocate_start_end = [&](UINT32 &seg_st, UINT32 &seg_end, UINT32 &mask_st, UINT32 &mask_end, bool is_row){
        GrowSeg gw_seg;
        this->GetGrowSegment(gw_seg, py, px, is_row);
        seg_st   = gw_seg.st;
        seg_end  = gw_seg.end;

        mask_end = mask.FindMaskBorderPoint(py, px, gw_seg.st, gw_seg.end, (is_row? 1 : m_wd));
        mask_st  = is_row? px : py;
        if(mask_end > mask_st)
            seg_end = seg_st>0? seg_st-1:seg_st;

        mask_st += 1;
        seg_st = mask_end;
    };

    auto ComputeBICcost = [&](UINT32 x0, UINT32 x1, UINT32 bias){
        return Log_LUT(x1-x0+bias+m_pParam->segGrow_shrk_bic_addi_len);
    };
    auto ComputeFitCost = [&](UINT32 py, UINT32 px, bool is_row){
        DpSeg dp_seg;
        m_pSegStock->GetDpSegment(dp_seg, py, px, is_row);
        UINT32 st = dp_seg.st, end = dp_seg.end;
        UINT32 bd_0 = is_row? m_borderH[py].first : m_borderV[px].first; 
        UINT32 bd_1 = is_row? m_borderH[py].second : m_borderV[px].second; 
        UINT32 line = is_row? py : px;
        if(bd_0 <= bd_1){
            m_pSegStock->GetDpSegment(dp_seg, line, bd_0, is_row);
            st  = min(st, dp_seg.st);
            m_pSegStock->GetDpSegment(dp_seg, line, bd_1, is_row);
            end = max(end, dp_seg.end);
        }
        float fit_cost = m_pSegStock->GetAllSegCost(line, st, end, is_row);
        fit_cost  += m_pParam->segGrow_shrk_fit_cost_penalty*(fit_cost>m_pParam->segGrow_shrk_fit_cost_thr);

        return fit_cost;
    };

    //find st/end for (py, px), and mask_st, mask_end inside mask from (py, px).
    UINT32 seg_st_A, seg_end_A, mask_end_A, mask_st_A;
    UINT32 seg_st_B, seg_end_B, mask_end_B, mask_st_B;

    // find st, end, cut position in extending & its orghogonal direction.
    if(ext_hor){
        relocate_start_end(seg_st_A, seg_end_A, mask_st_A, mask_end_A, true); // extending dir.
        relocate_start_end(seg_st_B, seg_end_B, mask_st_B, mask_end_B, false);// orthogonal dir. 
    }
    else{
        relocate_start_end(seg_st_B, seg_end_B, mask_st_B, mask_end_B, true); // orthogonal fir
        relocate_start_end(seg_st_A, seg_end_A, mask_st_A, mask_end_A, false);// extending dir
    }

    // compute BIC cost and FIT cost
    UINT32 len_A = seg_end_A > seg_st_A? seg_end_A-seg_st_A : seg_st_A-seg_end_A;
    float bic_A = ComputeBICcost(mask_st_A, seg_st_A, 1);
    bic_A += ComputeBICcost(mask_st_A, seg_end_A, UINT32(1+0.1*len_A));
    bic_A -= ComputeBICcost(mask_st_A, seg_st_A, 0);
    bic_A -= ComputeBICcost(mask_st_A, seg_end_A, UINT32(2+0.1*len_A));

    UINT32 len_B = seg_end_B > seg_st_B? seg_end_B-seg_st_B : seg_st_B-seg_end_B;
    float bic_B = ComputeBICcost(mask_st_B, seg_st_B, 1);
    bic_B += ComputeBICcost(mask_st_B, seg_end_B, UINT32(1+0.2*len_B));
    bic_B -= ComputeBICcost(mask_st_B, seg_st_B, 0);
    bic_B -= ComputeBICcost(mask_st_B, seg_end_B, UINT32(2+0.2*len_B));
    if (mask_st_B==mask_end_B)
        bic_B = 1e3;

    // Compute Fit cost of exist of the point.
    float fit_A_H = ComputeFitCost(py, px, true);
    float fit_A_V = ComputeFitCost(py, px, false);

    return -(fit_A_H+fit_A_V + m_pParam->segGrow_shrk_bic_alpha*(bic_A+bic_B));
}

void Segment_Grow::GrowingShrink(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor){
    auto ComputeCost = [&](UINT32 py, UINT32 px, auto &cost_heap){
        if((px > 0 && mask.GetData(py, px-1)>0 && px < m_wd-1 && mask.GetData(py, px+1))>0)
            return;
        else if(py > 0 && mask.GetData(py-1, px)>0 && py < m_ht-1 && mask.GetData(py+1, px)>0)
            return;
        else if (mask.GetData(py, px) == 2){
            float rm_cost = ComputeOneNodeCost(mask, py, px, ext_hor);
            Seed node(py, px, rm_cost);
            cost_heap.push(node);
        }
    };

    auto UpdateSegmentStock = [&](UINT32 py, UINT32 px, bool is_row){
        // update segment stock.
        GrowSeg gw_seg;
        this->GetGrowSegment(gw_seg, py, px, is_row);
        
        // Disable [st, end)
        this->DisableGrowSegment(gw_seg.id, is_row);
        
        // add new sub segment to segment stock.
        UINT32 line = is_row? py : px;
        UINT32  ptK = is_row? px : py;
        bool mask_on_end = is_row? mask.GetData(py, px+1) != ms_BG : mask.GetData(py+1, px)!=ms_BG;
        if(mask_on_end)
            // mask on end's side, [st, end] => [st, ptK) & [ptK, end)
            if(ptK > gw_seg.st)
                this->AddGrowSegment(line, gw_seg.st, ptK, is_row);
        else 
            // mask on st's side, segment is cutted to [st, px+1) & [px+1, end)
            if(ptK+1 < gw_seg.end)
                this->AddGrowSegment(line, ptK, gw_seg.end, is_row);
    };

    // starting from corner points on the boundary.
    priority_queue<Seed, vector<Seed>, SeedCmp> pix_seeds;
    for(UINT32 k=0; k<bdPair.size(); k++){
        if(mask.GetData(bdPair[k].first, bdPair[k].second) == ms_POS_FG)
            ComputeCost(bdPair[k].first, bdPair[k].second, pix_seeds);
    }

    // greedy shrink from pixels with smallest remove cost.
    while(!pix_seeds.empty()){
        Seed top_node = pix_seeds.top();
        pix_seeds.pop();
        
        // shrink back if satisfy the condition.
        UINT32 py = top_node.id0, px = top_node.id1;
        if(top_node.cost < m_pParam->segGrow_shrk_cost_thr){
            mask.SetData(ms_BG, py, px);
            
            if (py>0 && mask.GetData(py-1, px)==ms_POS_FG)
                ComputeCost(py-1, px, pix_seeds);
            if (py<m_ht-1 && mask.GetData(py+1, px)==ms_POS_FG)
                ComputeCost(py+1, px, pix_seeds);
            if (px>0 && mask.GetData(py, px-1)==ms_POS_FG)
                ComputeCost(py, px-1, pix_seeds);
            if (px<m_wd-1 && mask.GetData(py, px+1)==ms_POS_FG)
                ComputeCost(py, px+1, pix_seeds);
        }
        else{ // if the pixel is retained.
            UpdateSegmentStock(py, px, true);
            UpdateSegmentStock(py, px, false);
        }
    }
}


void Segment_Grow::GrowingExtend(CDataTempl<UINT32> &mask, vector<pair<UINT32, UINT32> > &bdPair, bool ext_hor){
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
    
    // extend along boundary pixels.
    ResetBorderHV();
    for(int k=0; k < pre_bds.size(); k++){
        UINT32 py   = pre_bds[k].first;
        UINT32 px   = pre_bds[k].second;
        UINT32 line = ext_hor? py : px;

        // if have been assigned a label.
        if(m_out_maskI.GetData(py, px) > 0)
            continue;
        
        // update (min,max) border for each row, column
        m_borderH[py].first  = px < m_borderH[py].first?  px : m_borderH[py].first;
        m_borderH[py].second = px > m_borderH[py].second? px : m_borderH[py].second;
        m_borderV[px].first  = py < m_borderV[px].first?  py : m_borderV[px].first;
        m_borderV[px].second = py > m_borderV[px].second? py : m_borderV[px].second;
        
        // update curMask, growing along whole segment where boundary locates on. 
        GrowSeg gw_seg;
        this->GetGrowSegment(gw_seg, py, px, ext_hor);
        if(gw_seg.valid == false)
            continue;
        for(int sk = gw_seg.st; sk < gw_seg.end; sk++){
            UINT32 mask_id =  ext_hor? mask.GetData(line, sk) : mask.GetData(sk, line);
            
            // if the pixel could be extended.
            if(mask_id == ms_BG)
                if(ext_hor)
                    mask.SetData(ms_POS_FG, line, sk);
                else
                    mask.SetData(ms_POS_FG, sk, line);
        }

        // set edge pixel to bdPair, used as seed for shrink processing.
        Insert2bdPair(line, gw_seg.st);
        if(gw_seg.end-1 != gw_seg.st)
            Insert2bdPair(line, gw_seg.end-1);
        
    }
}

void Segment_Grow::GrowingFromASegment(UINT32 grow_seed_id, UINT32 &propId){
    
    // initial a mask from seed.   
    GrowSeg gw_seg;
    this->GetGrowSegment(gw_seg, UINT32(grow_seed_id), true);
    CDataTempl<UINT32> curMask(m_ht, m_wd);
    curMask.ResetBulkData(ms_FG, gw_seg.line, 1, gw_seg.st, gw_seg.end-gw_seg.st);

    // Growing from seed looply.
    bool ext_hor = false;
    UINT32 grow_tot = 0, grow_1step = 30; 
    while(grow_1step>0){
        // step 1:: extend along boundary pixels based on orogonal segments.
        vector<pair<UINT32, UINT32> > bdPair;
        GrowingExtend(curMask, bdPair, ext_hor);
#ifdef DEBUG_SEGMENT_GROW
        WriteToCSV(curMask, "./output/test_extend.csv");
        string py_command = "python pyShow.py";
        system(py_command.c_str());
#endif
        // step 2:: starting from corner points, shrink back pixel by pixel.
        GrowingShrink(curMask, bdPair, ext_hor);
#ifdef DEBUG_SEGMENT_GROW
         WriteToCSV(curMask, "./output/test_shrink.csv");
#endif

        // step 3:: retained growing pixels merge to the group, used as seed for next growing loop.
        grow_1step = curMask.ReplaceByValue(ms_POS_FG, ms_FG);
        grow_tot  += grow_1step;
        ext_hor    = !ext_hor;
    }

    if(grow_tot > m_pParam->segGrow_proposal_size_thr){
        cout<<propId<<" ::  total grow size is ... "<<grow_tot<<endl;
        curMask.ModifyMaskOnNonZeros(m_out_maskI, propId);
        propId += 1;
    }
}


void Segment_Grow::ImagePartition(CDataTempl<UINT32> &sem_bgI){
    UINT32 propId = 1;
    sem_bgI.ModifyMaskOnNonZeros(m_out_maskI, propId);
    propId  += 1;
    
    while(m_grow_seg_seeds.size()>0){
        // pop out a seed, and do growing from it if valid.
        Seed top_node(0,0,0);
        top_node = m_grow_seg_seeds.top();
        m_grow_seg_seeds.pop();

        // Grow from the seed if it is valid 
        if(this->GrowSegmentIsValid(top_node.id0, true)){
            this->DisableGrowSegment(top_node.id0, true);
            GrowingFromASegment(top_node.id0, propId);
            /* 
            GrowSeg gw_seg;
            m_pSegStock->GetGrowSegment(gw_seg, UINT32(top_node.id0), true);
            float acost = m_pSegStock->GetAllSegCost(gw_seg.line, gw_seg.st, gw_seg.end, true); 
            m_out_maskI.ResetBulkData(acost==gw_seg.fit_err, gw_seg.line, 1, gw_seg.st, gw_seg.end-gw_seg.st); 
            */
        }
    }
    cout<<"Segment Grow Donw!."<<endl;
    cout<<"*******************************************"<<endl;
}

#endif
