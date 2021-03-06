#ifndef SEGMENT_GROW_HXX
#define SEGMENT_GROW_HXX

#include "utils/read_write_img.hxx"
#include "SegmentFitting.hxx"
#include "SegmentStock.hxx"

using namespace std;
 
typedef struct Grow_Segment{
    int id;
    int len;
    int y0, x0;
    int y1, x1;
    double  fit_err;
    vector<double> sem_score;
    double  sem_cost;
    bool   valid;
    Grow_Segment(){
        id = -1; len = 0;
        y0 = 0;   x0 = 0;
        y1 = 0;   x1 = 0;
        valid    = false;
        fit_err  = 1e3;
        sem_cost = 0;
    }
    Grow_Segment(int z, int y, int a, int b, int c, int d, bool flag=true){
        id = z;  len = y;
        y0 = a;   x0 = b;
        y1 = c;   x1 = d;
        valid    = flag;
        fit_err  = 1e3;
        sem_cost = 0;
    }
    
    Grow_Segment(int z, DpSeg &dp_seg, bool flag=true){
        id = z;         len = dp_seg.len;
        y0 = dp_seg.y0;  x0 = dp_seg.x0;
        y1 = dp_seg.y1;  x1 = dp_seg.x1;
        valid   = flag;
        fit_err = dp_seg.seg_info.fit_err;
        sem_score = dp_seg.seg_info.sem_score;

        double max_sem_score = 1e-5;
        for(int k=0; k < sem_score.size(); k++){
            if(max_sem_score < sem_score[k])
                max_sem_score = sem_score[k];
        }
        sem_cost = -log(max_sem_score/(1-max_sem_score+1e-5));//_NegativeLog(max_sem_score);
    }
} GrowSeg; // [st, end)

enum Mask_type {ms_BG=0, ms_FG, ms_POS_FG};

/*
 * Class: Segment_Grow.
 *        Based on segment, do region growing.
 *
 *   API: ImagePartition()
 *        GetFinalResult()
 */

class Segment_Grow{
protected:
    // variables exist in the whole life.
    bool   m_row_as_seed;  // each growing depends on row segments if true, otherwise col segs.
    CDataTempl<double>        * m_pSemMat;
    const CDataTempl<int>  * m_pSem_bg;
    Segment_Stock *            m_pSegStock; // segment informations 
   
    // generated info.
    int m_ht, m_wd;
    int m_num_sem;
    CDataTempl<int>   m_grow_segInfo;
    vector<GrowSeg>      m_grow_seg;
    priority_queue<Seed_1D, vector<Seed_1D>, SeedCmp_1D> m_grow_seg_seeds;
    
    // output
    CDataTempl<double>    m_out_debugI;
    CDataTempl<int>   m_out_maskI; // segment grow mask. 0-unprocessed data, 1-background. >1 - grown region label.
    
    // tmp variables for each shrink process in growing.
    vector<pair<int, int> > m_borderH; // used to record the border pixel of current mask in each line
    vector<pair<int, int> > m_borderV; // ... in each column

    // parameters.
    const GlbParam *m_pParam;

public:
    Segment_Grow(bool row_as_seed, const auto *pSem_bg, auto *pSemMat, Segment_Stock * pSegStock, const GlbParam *pParam){
        m_row_as_seed = row_as_seed;
        m_pSem_bg     = pSem_bg;
        m_pSemMat     = pSemMat;
        m_pSegStock   = pSegStock;
        m_pParam      = pParam;
        
        m_ht      = m_pSemMat->GetYDim();
        m_wd      = m_pSemMat->GetXDim();
        m_num_sem = m_pSemMat->GetZDim();
        m_out_debugI.Init(m_ht, m_wd, 3);
        m_out_maskI.Init(m_ht, m_wd);
        m_grow_segInfo.Init(m_ht, m_wd, e_seg_type_num);
        
        InitialSetGrowSegments();
    }
    CDataTempl<int>& GetFinalResult();
    CDataTempl<double>& GetDebugInformation();
    
    void ImagePartition();
    
protected: 
    // operations on "grow segments"
    double ComputeSeedCost(GrowSeg &gw_seg);
    void InitialSetGrowSegments();
    void GetGrowSegmentById(GrowSeg &gw_seg, int id);
    void GetGrowSegmentByCoord(GrowSeg &gw_seg, int py, int px, SegType s_type);
    bool IsGrowSegmentValid(int id);
    void DisableGrowSegment(int id);
    void AddGrowSegment(GrowSeg &ori_seg, int len, int y0, int x0, int y1, int x1);

    // Generate a label image with segment growing.
    void GrowingFromASegment(int grow_seed_id, int &propId, bool is_row);
    void GrowingExtend(CDataTempl<int> &mask, vector<pair<int, int> > &bdPair, bool ext_hor);
    void GrowingShrink(CDataTempl<int> &mask, vector<pair<int, int> > &bdPair, bool ext_hor);
    double ComputeOneNodeCost(CDataTempl<int> &mask, int py, int px, bool ext_hor, int &mask_end_A, int &mask_end_B);

    void ResetBorderHV();
};

CDataTempl<int>& Segment_Grow::GetFinalResult(){
    auto ProcessOnePoint = [&](SINT32 y, SINT32 x, auto &oriI, auto &distI){
        // traverse all neighboor pixels, find the min distance to label pixels.
        int min_nei_dist = m_ht+m_wd;
        for(SINT32 sy=-1; sy<= 0; sy++){
            for(SINT32 sx=-1; sx <=1; sx ++){
                if(distI.GetData(y+sy, x+sx)>0)
                    min_nei_dist = min(min_nei_dist, distI.GetData(y+sy, x+sx));
            }
        }
        min_nei_dist = min_nei_dist==m_ht+m_wd? 2 : min_nei_dist;
        
        // go through ring with distance min_nei_dist-1, min_nei_dist, min_nei_dist+1. 
        // to find which label should assign to this pixel.
        for(SINT32 sk=-1; sk<=1; sk++){
            SINT32 win    = min_nei_dist+sk;
            // left line
            for(SINT32 sy = -win; sy <= win; sy ++){
                if(oriI.GetData(y+sy, x-win)>0){
                    m_out_maskI.SetData(oriI.GetData(y+sy, x-win), y, x);
                    distI.SetData(win, y, x);
                    return;
                }
            }
            // upper line
            for(SINT32 sx = -win; sx <= win; sx ++){
                if(oriI.GetData(y-win, x+sx)>0){
                    m_out_maskI.SetData(oriI.GetData(y-win, x+sx), y, x);
                    distI.SetData(win, y, x);
                    return;
                }
            }
            // right line
            for(SINT32 sy = -win; sy <= win; sy ++){
                if(oriI.GetData(y+sy, x+win)>0){
                    m_out_maskI.SetData(oriI.GetData(y+sy, x+win), y, x);
                    distI.SetData(win, y, x);
                    return;
                }
            }
            // bottom line
            for(SINT32 sx = -win; sx <= win; sx ++){
                if(oriI.GetData(y+win, x+sx)>0){
                    m_out_maskI.SetData(oriI.GetData(y+win, x+sx), y, x);
                    distI.SetData(win, y, x);
                    return;
                }
            }
        } 
    };
    if(m_pParam->segGrow_rm_label0){
        CDataTempl<int> distI(m_ht, m_wd);
        CDataTempl<int> ori_labelI(m_ht, m_wd);
        ori_labelI.Copy(m_out_maskI);

        // clockwise order
        for(SINT32 y = 0; y < m_ht; y++){
            for(SINT32 x = 0; x < m_wd; x++){
                if(ori_labelI.GetData(y, x) > 0)
                    continue;
                else
                    ProcessOnePoint(y, x, ori_labelI, distI);
            }
        }
        // inverse order. 
        for(SINT32 y = m_ht-1; y >= 0; y--){
            for(SINT32 x = m_wd-1; x >= 0; x--){
                if(m_out_maskI.GetData(y, x) > 0)
                    continue;
                else
                    ProcessOnePoint(y, x, ori_labelI, distI);
            }
        }
    }

    return m_out_maskI;
}
CDataTempl<double>& Segment_Grow::GetDebugInformation(){
    int num_Seg = m_grow_seg.size();
    for(int k =0; k < num_Seg; k++){
        GrowSeg gw_seg = m_grow_seg[k];
        if(gw_seg.y0 == gw_seg.y1)
            m_out_debugI.ResetBulkData(gw_seg.fit_err, gw_seg.y0, 1, gw_seg.x0, gw_seg.x1-gw_seg.x0); 
    }
    return m_out_debugI;
}

// function for managing growing segments.
double Segment_Grow::ComputeSeedCost(GrowSeg &gw_seg){
    double sem_cost = m_pParam->segGrow_seed_sem_alpha * gw_seg.sem_cost;
    double bic_cost = m_pParam->segGrow_seed_bic_alpha * (1/log(m_pParam->segGrow_seed_bic_scale*gw_seg.len + 2));
    
    return (gw_seg.fit_err + bic_cost + sem_cost);
}

void Segment_Grow::InitialSetGrowSegments(){
    int num_dpSeg = m_pSegStock->GetDpSegmentSize();
    for(int k = 0; k < num_dpSeg; k++){
        DpSeg dp_seg;
        m_pSegStock->GetDpSegmentById(dp_seg, k);

        // define grow segment from dp segment. set it invalid if on semantic background.
        GrowSeg gw_seg(k, dp_seg, true);
        if(m_pSem_bg->GetData((gw_seg.y0+gw_seg.y1)/2, (gw_seg.x0+gw_seg.x1)/2) == 1){
            gw_seg.valid=false;
            gw_seg.fit_err = 1e3;
        }
        
        // special dealing about point_1 in segment. since dp segment is: [pt0, pt1], grow segment is [pt0, pt1)
        if(gw_seg.y0 == gw_seg.y1 && gw_seg.x1 == m_wd-1)
            gw_seg.x1 = m_wd;
        else if(gw_seg.x0 == gw_seg.x1 && gw_seg.y1 == m_ht-1)
            gw_seg.y1 = m_ht;
        m_grow_seg.push_back(gw_seg);


        if(gw_seg.y1==gw_seg.y0){
            m_grow_segInfo.ResetBulkData(k, gw_seg.y0, 1, gw_seg.x0, gw_seg.x1-gw_seg.x0, e_seg_h, 1);
            // if the segment is valid, add into segment seed.
            if(m_row_as_seed==true && gw_seg.valid){
                double cost = ComputeSeedCost(gw_seg);
                Seed_1D seg_seed(k, cost);
                m_grow_seg_seeds.push(seg_seed);
            }
        }
        else{
            m_grow_segInfo.ResetBulkData(k, gw_seg.y0, gw_seg.y1-gw_seg.y0, gw_seg.x0, 1, e_seg_v, 1);
            // if the segment is valid, add into segment seed.
            if(m_row_as_seed==false && gw_seg.valid){
                double cost = ComputeSeedCost(gw_seg);
                Seed_1D seg_seed(k, cost);
                m_grow_seg_seeds.push(seg_seed);
            }
        }
    }
}

void Segment_Grow::GetGrowSegmentById(GrowSeg &gw_seg, int id){
    gw_seg = m_grow_seg[id];
}

void Segment_Grow::GetGrowSegmentByCoord(GrowSeg &gw_seg, int py, int px, SegType s_type){
    int id = m_grow_segInfo.GetData(py, px, s_type);
    this->GetGrowSegmentById(gw_seg, id);
}

bool Segment_Grow::IsGrowSegmentValid(int id){
    return m_grow_seg[id].valid;
}

void Segment_Grow::DisableGrowSegment(int id){
    m_grow_seg[id].valid = false;
}

void Segment_Grow::AddGrowSegment(GrowSeg &ori_seg, int len, int y0, int x0, int y1, int x1){
    // adding of gw_segment.
    int seg_id    = m_grow_seg.size();
    GrowSeg gw_seg(seg_id, len, y0, x0, y1, x1, true);
    gw_seg.fit_err   = ori_seg.fit_err;
    gw_seg.sem_score = ori_seg.sem_score;
    gw_seg.sem_cost  = ori_seg.sem_cost;
    m_grow_seg.push_back(gw_seg);

    if(y0 == y1){
        m_grow_segInfo.ResetBulkData(seg_id, y0, 1, x0, x1-x0, e_seg_h, 1);
        // add of segment seed.
        if(m_row_as_seed){
            double cost = ComputeSeedCost(gw_seg);
            Seed_1D seg_seed(seg_id, cost);
            m_grow_seg_seeds.push(seg_seed);
        }
    }
    else{
        m_grow_segInfo.ResetBulkData(seg_id, y0, y1-y0, x0, 1, e_seg_v, 1);
        // add of segment seed.
        if(!m_row_as_seed){
            double cost = ComputeSeedCost(gw_seg);
            Seed_1D seg_seed(seg_id, cost);
            m_grow_seg_seeds.push(seg_seed);
        }
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
double Segment_Grow::ComputeOneNodeCost(CDataTempl<int> &mask, int py, int px, bool ext_hor, int &mask_end_A, int &mask_end_B){
    auto relocate_start_end = [&](int &seg_st, int &seg_end, int &mask_st, int &mask_end, bool is_row){
        GrowSeg gw_seg;
        if(is_row){
            this->GetGrowSegmentByCoord(gw_seg, py, px, e_seg_h);
            seg_st   = gw_seg.x0;
            seg_end  = gw_seg.x1;
        }
        else{
            this->GetGrowSegmentByCoord(gw_seg, py, px, e_seg_v);
            seg_st   = gw_seg.y0;
            seg_end  = gw_seg.y1;
        }

        mask_end = mask.FindMaskBorderPoint(py, px, seg_st, seg_end, (is_row? 1 : m_wd));
        mask_st  = is_row? px : py;
        if(mask_end > mask_st){
            seg_end = seg_st>0? seg_st-1:seg_st;
            mask_st = mask_st>0? mask_st-1 : mask_st;
        }
        seg_st = mask_end;
    };

    auto ComputeBICcost = [&](int x0, int x1, double bias){
        int len = x1>x0? x1-x0 : x0-x1;
        double cost = log(len+bias+m_pParam->segGrow_shrk_bic_addi_len);
        return cost;
    };
    auto ComputeFitCost = [&](int py, int px){
        int st, end;
        DpSeg dp_seg;
        double fit_cost;

        // H direction
        int bd_0 = m_borderH[py].first;
        int bd_1 = m_borderH[py].second;
        if(bd_0 <= bd_1){
            m_pSegStock->GetDpSegmentByCoord(dp_seg, py, bd_0, e_seg_h);
            st = dp_seg.x0;
            m_pSegStock->GetDpSegmentByCoord(dp_seg, py, bd_1, e_seg_h);
            end = dp_seg.x1;
        }
        else{
            m_pSegStock->GetDpSegmentByCoord(dp_seg, py, px, e_seg_h);
            st = dp_seg.x0;
            end = dp_seg.x1;
        }
        fit_cost = m_pSegStock->GetAllSegFitError(py, st, py, end);

        // V direction
        bd_0 = m_borderV[px].first;
        bd_1 = m_borderV[px].second;
        if(bd_0 <= bd_1){
            m_pSegStock->GetDpSegmentByCoord(dp_seg, bd_0, px, e_seg_v);
            st = dp_seg.y0;
            m_pSegStock->GetDpSegmentByCoord(dp_seg, bd_1, px, e_seg_v);
            end = dp_seg.y1;
        }
        else{
            m_pSegStock->GetDpSegmentByCoord(dp_seg, py, px, e_seg_v);
            st = dp_seg.y0;
            end = dp_seg.y1;
        }
        fit_cost += m_pSegStock->GetAllSegFitError(st, px, end, px);
        fit_cost += m_pParam->segGrow_shrk_fit_cost_penalty*(fit_cost>m_pParam->segGrow_shrk_fit_cost_thr);

        return fit_cost;
    };

    //find st/end for (py, px), and mask_st, mask_end inside mask from (py, px).
    int seg_st_A, seg_end_A, mask_st_A; // mask_end_A 
    int seg_st_B, seg_end_B, mask_st_B; // mask_end_B

    // find st, end, cut position in extending & its orghogonal direction.
    if(ext_hor){
        relocate_start_end(seg_st_A, seg_end_A, mask_st_A, mask_end_A, true); // extending dir.
        relocate_start_end(seg_st_B, seg_end_B, mask_st_B, mask_end_B, false);// orthogonal dir. 
    }
    else{
        relocate_start_end(seg_st_B, seg_end_B, mask_st_B, mask_end_B, true); // orthogonal fir
        relocate_start_end(seg_st_A, seg_end_A, mask_st_A, mask_end_A, false);// extending dir
    }

    // compute BIC cost and FIT cost for existance of the point. 
    // A is the extending direction. B is the orthogonal direction.
    int len_A = seg_end_A > seg_st_A? seg_end_A-seg_st_A : seg_st_A-seg_end_A;
    double bic_A = ComputeBICcost(mask_st_A, seg_st_A, 1);
    bic_A += ComputeBICcost(mask_st_A, seg_end_A, 1+0.1*len_A);
    bic_A -= ComputeBICcost(mask_st_A, seg_st_A, 0);
    bic_A -= ComputeBICcost(mask_st_A, seg_end_A, 2+0.1*len_A);

    int len_B = seg_end_B > seg_st_B? seg_end_B-seg_st_B : seg_st_B-seg_end_B;
    double bic_B = ComputeBICcost(mask_st_B, seg_st_B, 1);
    bic_B += ComputeBICcost(mask_st_B, seg_end_B, 1+0.2*len_B);
    bic_B -= ComputeBICcost(mask_st_B, seg_st_B, 0);
    bic_B -= ComputeBICcost(mask_st_B, seg_end_B, 2+0.2*len_B);
    if (mask_st_B==mask_end_B)
        bic_B = 1e3;

    // Compute Fit cost of existance of the point.
    double fit_A = ComputeFitCost(py, px);
    double exist_cost = fit_A + m_pParam->segGrow_shrk_bic_alpha*(bic_A+bic_B);
    
    m_out_debugI.SetData(exist_cost, py, px, 0);
    m_out_debugI.SetData(fit_A, py, px, 1);
    m_out_debugI.SetData(bic_A+bic_B, py, px, 2);

    return (-exist_cost);
}

void Segment_Grow::GrowingShrink(CDataTempl<int> &mask, vector<pair<int, int> > &bdPair, bool ext_hor){
    auto ComputeCost = [&](int py, int px, auto &cost_heap, auto &&seeds_map){
        // compute cost of removing a corner pixel, add it to a heap.
        if((px > 0 && mask.GetData(py, px-1)>0 && px < m_wd-1 && mask.GetData(py, px+1))>0)
            return;
        else if(py > 0 && mask.GetData(py-1, px)>0 && py < m_ht-1 && mask.GetData(py+1, px)>0)
            return;
        else if (mask.GetData(py, px) == 2){
            int mask_end_ext, mask_end_orth;
            double rm_cost = ComputeOneNodeCost(mask, py, px, ext_hor, mask_end_ext, mask_end_orth);
            Seed node(py, px, rm_cost, mask_end_ext, mask_end_orth);
            cost_heap.push(node);
            Mkey_2D node2(py, px);
            seeds_map[node2] = rm_cost;
        }
    };

    auto UpdateSegmentStock = [&](int py, int px, int mask_end, SegType s_type){
        // update segment stock.
        GrowSeg gw_seg;
        this->GetGrowSegmentByCoord(gw_seg, py, px, s_type);
        if(gw_seg.valid == false)
            return;
        
        // Disable [st, end)
        this->DisableGrowSegment(gw_seg.id);
        
        // add new sub segment to segment stock.
        int ptK = s_type==e_seg_h? px : py;
        if(mask_end >= ptK){
            // mask on end's side, [st, end] => [st, ptK) & [ptK, mask_end] & [mask_end+1, end)
            if(s_type == e_seg_h){
                if(ptK > gw_seg.x0)
                    this->AddGrowSegment(gw_seg, px-gw_seg.x0, py, gw_seg.x0, py, ptK);
                if(mask_end+1 < gw_seg.x1)
                    this->AddGrowSegment(gw_seg, gw_seg.x1-mask_end-1, py, mask_end+1, py, gw_seg.x1);
            }
            else{
                if(ptK > gw_seg.y0)
                    this->AddGrowSegment(gw_seg, py-gw_seg.y0, gw_seg.y0, px, ptK, px);
                if(mask_end+1 < gw_seg.y1)
                    this->AddGrowSegment(gw_seg, gw_seg.y1-mask_end-1, mask_end+1, px, gw_seg.y1, px);
            }
        }
        else{
            // mask on st's side, segment is cutted to [st, mask_end) &[mask_end, ptK] & [ptK+1, end)
            if(s_type == e_seg_h){
                if(mask_end > gw_seg.x0)
                    this->AddGrowSegment(gw_seg, mask_end-gw_seg.x0, py, gw_seg.x0, py, mask_end);
                if(ptK+1 < gw_seg.x1)
                    this->AddGrowSegment(gw_seg, gw_seg.x1-ptK-1, py, ptK+1, py, gw_seg.x1);
            }
            else{
                if(mask_end > gw_seg.y0)
                    this->AddGrowSegment(gw_seg, ptK-gw_seg.y0, gw_seg.y0, px, mask_end, px);
                if(ptK+1 < gw_seg.y1)
                    this->AddGrowSegment(gw_seg, gw_seg.y1-ptK-1, ptK+1, px, gw_seg.y1, px);
            } 
        }
    };

    // Main Process :: starting from corner points on the boundary.
    priority_queue<Seed, vector<Seed>, SeedCmp> pix_seeds;
    map<Mkey_2D, double, MKey2DCmp> seeds_map;
    for(int k=0; k<bdPair.size(); k++){
        ComputeCost(bdPair[k].first, bdPair[k].second, pix_seeds, seeds_map);
    }

    // greedy shrink from pixels with smallest remove cost.
    while(!pix_seeds.empty()){
        Seed top_node = pix_seeds.top();
        pix_seeds.pop();
        
        // check if it is the latest status.
        int py = top_node.id0, px = top_node.id1;
        Mkey_2D node2(py, px);
        if(seeds_map[node2] != top_node.cost || mask.GetData(py, px)!=ms_POS_FG)
            continue;
        
        // shrink back if satisfy the condition.
        if(top_node.cost < m_pParam->segGrow_shrk_cost_thr){
            mask.SetData(ms_BG, py, px);
            
            if (py>0 && mask.GetData(py-1, px)==ms_POS_FG)
                ComputeCost(py-1, px, pix_seeds, seeds_map);
            if (py<m_ht-1 && mask.GetData(py+1, px)==ms_POS_FG)
                ComputeCost(py+1, px, pix_seeds, seeds_map);
            if (px>0 && mask.GetData(py, px-1)==ms_POS_FG)
                ComputeCost(py, px-1, pix_seeds, seeds_map);
            if (px<m_wd-1 && mask.GetData(py, px+1)==ms_POS_FG)
                ComputeCost(py, px+1, pix_seeds, seeds_map);
        }
        else{ // if the pixel is retained.
            UpdateSegmentStock(py, px, (ext_hor? top_node.tmp0 : top_node.tmp1), e_seg_h);
            UpdateSegmentStock(py, px, (ext_hor? top_node.tmp1 : top_node.tmp0), e_seg_v);
        }
    }
}

void Segment_Grow::GrowingExtend(CDataTempl<int> &mask, vector<pair<int, int> > &bdPair, bool ext_hor){
    // starting process.
    vector<pair<int, int> > pre_bds;
    mask.FindBoundaryOnMask(pre_bds, 1);
    vector<double> exp_sem_score(m_num_sem, 0);
    m_pSemMat->MeanZ(mask, int(1), exp_sem_score);
    
    // extend along boundary pixels.
    ResetBorderHV();
    for(int k=0; k < pre_bds.size(); k++){
        int py   = pre_bds[k].first;
        int px   = pre_bds[k].second;
        
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
        this->GetGrowSegmentByCoord(gw_seg, py, px, (ext_hor? e_seg_h : e_seg_v));
        if(gw_seg.valid == false || _ChiDifference(gw_seg.sem_score, exp_sem_score)>m_pParam->segGrow_extd_semdiff_thr)
            continue;

        int st  = ext_hor? gw_seg.x0 : gw_seg.y0;
        int end = ext_hor? gw_seg.x1 : gw_seg.y1;
        for(int sk = st; sk < end; sk++){
            int mask_id  =  ext_hor? mask.GetData(gw_seg.y0, sk) : mask.GetData(sk, gw_seg.x0);
            int glb_flag =  ext_hor? m_out_maskI.GetData(gw_seg.y0, sk) : m_out_maskI.GetData(sk, gw_seg.x0);
            
            // if the pixel could be extended.
            if(mask_id == ms_BG && glb_flag==0){
                if(ext_hor)
                    mask.SetData(ms_POS_FG, gw_seg.y0, sk);
                else
                    mask.SetData(ms_POS_FG, sk, gw_seg.x0);
            }
        }

        // set edge pixel to bdPair, used as seed for shrink processing.
        int mask_st_id =  mask.GetData(gw_seg.y0, gw_seg.x0);
        if(mask_st_id == ms_POS_FG)
            bdPair.push_back(make_pair(gw_seg.y0, gw_seg.x0));
        
        int mask_end_id = ext_hor? mask.GetData(gw_seg.y1, gw_seg.x1-1) : mask.GetData(gw_seg.y1-1, gw_seg.x1);
        if(mask_end_id == ms_POS_FG){
            if(ext_hor)
                bdPair.push_back(make_pair(gw_seg.y1, gw_seg.x1-1));
            else
                bdPair.push_back(make_pair(gw_seg.y1-1, gw_seg.x1));
        }
    }
}

void Segment_Grow::GrowingFromASegment(int grow_seed_id, int &propId, bool is_row){
    
    // initial a mask from seed, and if any pixel on the seed is processed, return. 
    GrowSeg gw_seg;
    this->GetGrowSegmentById(gw_seg, int(grow_seed_id));
    for(int k=0; k<gw_seg.len; k++){
        int py = gw_seg.y0==gw_seg.y1? gw_seg.y0 : gw_seg.y0+k;
        int px = gw_seg.x0==gw_seg.x1? gw_seg.x0 : gw_seg.x0+k;
        if(m_out_maskI.GetData(py, px) > 0)
            return;
    }

    // Growing from seed looply.
    CDataTempl<int> curMask(m_ht, m_wd);
    if(is_row)
        curMask.ResetBulkData(ms_FG, gw_seg.y0, 1, gw_seg.x0, gw_seg.x1-gw_seg.x0);
    else
        curMask.ResetBulkData(ms_FG, gw_seg.y0, gw_seg.y1-gw_seg.y0, gw_seg.x0, 1);

    bool ext_hor = is_row? false : true;
    int grow_tot = 0, grow_1step = is_row? gw_seg.x1-gw_seg.x0:gw_seg.y1-gw_seg.y0; 
    while(grow_1step>0){
        
        // step 1:: extend along boundary pixels based on orogonal segments.
        vector<pair<int, int> > bdPair;
        GrowingExtend(curMask, bdPair, ext_hor);
#ifdef DEBUG_SEGMENT_GROW_STEP
        WriteToCSV(curMask, "./output/test_extend.csv");
#endif
        // step 2:: starting from corner points, shrink back pixel by pixel.
        GrowingShrink(curMask, bdPair, ext_hor);
        
        // step 3:: retained growing pixels merge to the group, used as seed for next growing loop.
        grow_1step = curMask.ReplaceByValue(ms_POS_FG, ms_FG);
#ifdef DEBUG_SEGMENT_GROW_STEP
        WriteToCSV(curMask, "./output/test_shrink.csv");
        if(true){//propId == 9 || propId == 11 || propId==18){
            string py_command = "python pyShow.py";
            system(py_command.c_str());
        }
#endif
        grow_tot  += grow_1step;
        ext_hor    = !ext_hor;
    }

    if(grow_tot > m_pParam->segGrow_proposal_size_thr){
        if (OPEN_DEBUG)
            cout<<propId<<" ::  total grow size is ... "<<grow_tot<<endl;
        m_out_maskI.ModifyMaskOnNonZeros(curMask, propId);
        propId += 1;
        
#ifdef DEBUG_SEGMENT_GROW_STEP
        WriteToCSV(m_out_maskI, "./output/test.csv", 1);
        string py_command = "python pyShow.py";
        system(py_command.c_str());
#endif
    }
}


void Segment_Grow::ImagePartition(){
    int propId = 1;
    m_out_maskI.ModifyMaskOnNonZeros(*m_pSem_bg, propId);
    propId  += 1;
    
    while(m_grow_seg_seeds.size()>0){
        // pop out a seed, and do growing from it if valid.
        Seed_1D top_node(0,0.0);
        top_node = m_grow_seg_seeds.top();
        m_grow_seg_seeds.pop();
        
        // Grow from the seed if it is valid 
        if(IsGrowSegmentValid(top_node.id0)){
            this->DisableGrowSegment(top_node.id0);
            GrowingFromASegment(top_node.id0, propId, m_row_as_seed);
        }
    }
    if (OPEN_DEBUG){
        cout<<"Segment Grow Donw!."<<endl;
        cout<<"*******************************************"<<endl;
    }
}

#endif
