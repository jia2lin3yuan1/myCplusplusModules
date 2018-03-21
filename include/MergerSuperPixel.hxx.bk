#ifndef _MERGER_SUPER_PIXEL_HXX
#define _MERGER_SUPER_PIXEL_HXX

/*
 * Class: SuperPixelMerger.
 *    used to merge super pixels based on fitting error from distance map. It's also based on the Bayesian Information Criterion, which is used to balance the size and fitting error.
 *
 *   API: AssignInputLabel()
 *        CreateGraphFromLabelI()
 *        ComputeGraphWeights()
 *        Merger()
 *
 *        GetDebugImage()
 *        AssignOutputLabel()
 */


#include "utils/LogLUT.hxx"
#include "utils/graph.hxx"
#include "SegmentGrow.hxx"

typedef struct LineBorder{
    UINT32 minK;
    UINT32 maxK;
    UINT32 size;
    LineBorder(UINT32 minV=1e9, UINT32 maxV=0, UINT32 s=0){
        minK = minV; maxK = maxV; size = s;
    }
}LineBD;

typedef struct BorderInfo{
    // bounding box, <y0, x0, y1, x1>.
    UINT32 bbox[4];

    vector<LineBD> border_h; // <minH, maxH>
    vector<LineBD> border_v; // <minV, maxV>
    
    BorderInfo(){
        bbox[0] = UINT_MAX;
        bbox[1] = UINT_MAX;
        bbox[2] = 0;
        bbox[3] = 0;
    }
    
    void ResizeBorderHV(UINT32 ht, UINT32 wd){
        border_h.clear();
        LineBD h_linebd(ht, 0, 0);
        border_h.assign(bbox[2]-bbox[0]+1, h_linebd);
        
        border_v.clear();
        LineBD v_linebd(ht, 0, 0);
        border_v.assign(bbox[3]-bbox[1]+1, v_linebd);
    }
    void ClearBorderHV(){
        border_h.clear();
        border_h.shrink_to_fit();
        border_v.clear();
        border_v.shrink_to_fit();
    }

}Border;

class DistEdge:public Edge{
public:
    Border border;

    float mergecost;
    vector<float> new_sem_score;
    float new_fit_cost;
    float new_bic_cost;
    float new_perimeter;
    
    // functions
    DistEdge(UINT32 s1=0, UINT32 s2=0, float edge=0):Edge(s1, s2, edge),border(){
        new_fit_cost = 0.0;
        new_bic_cost = 0.0;
        mergecost    = 0.0;
        new_perimeter= 0.0;
    }
};

class DistSuperPixel:public Supix{
public:
    Border border;
    vector<float> sem_score;
    float fit_cost;
    float bic_cost;
    float perimeter;

    // functions
    DistSuperPixel():Supix(),border(){
        fit_cost  = 0.0;
        bic_cost  = 0.0;
        perimeter = 0.0;
    }
};

class SuperPixelMerger:public Graph<DistSuperPixel, DistEdge, BndryPix>{
protected:
    // inpput variables.
    CDataTempl<float> *m_pSemMat;
    Segment_Stock     *m_pSegStock;

    // generated variables
    UINT32  m_num_sem;
    priority_queue<Seed_1D, vector<Seed_1D>, SeedCmp_1D> m_merge_seeds;

    // Parameter.
    const GlbParam *m_pParam;

public:
    SuperPixelMerger(CDataTempl<float> *pSemMat, Segment_Stock *pSegStock, const GlbParam *pParam):Graph(pSemMat->GetYDim(), pSemMat->GetXDim()){
        m_pSemMat   = pSemMat;
        m_pSegStock = pSegStock;
        m_pParam    = pParam;

        m_num_sem = m_pSemMat->GetZDim();
    }
    
    // virtual function from Graph
    void UpdateSuperPixel(UINT32 sup, UINT32 edge);
    void ComputeGraphWeights();
    void ComputeEdgeWeights(UINT32 edge);
    

    // function working on adding member variables on Node and Edge.
    void ComputeSuperPixelCost(UINT32 sup);
    float ComputeFitCost(UINT32 y0, UINT32 x0, UINT32 y1, UINT32 x1, UINT32 size);
    float ComputeBICcost(UINT32 numPix);
    float ComputeSemanticDifference(UINT32 sup0, UINT32 sup1);

    // Merge operations
    void Merger();

    void GetDebugImage(CDataTempl<float> &debugI, UINT32 mode=0);
    void PrintOutInformation();

};

void SuperPixelMerger::PrintOutInformation(){
    /*
    cout<<"** super pixels: "<<m_supixs.size()<<endl;
    for(auto it=m_supixs.begin(); it!= m_supixs.end(); it++){
        auto &supix = it->second;
        cout<<"*** No. "<<it->first<<",  size: " << supix.pixs.size()<<endl;
        cout<<"  bbox is: "<<supix.border.bbox[0]<<", "<<supix.border.bbox[1]<<", "<<supix.border.bbox[2]<<", "<<supix.border.bbox[3]<<endl;
        cout<<"  semantic score is: "<<endl;
        for(auto sem: supix.sem_score){
            cout << setprecision(5)<<sem<<", ";
        }
        cout << endl<<endl;
    }
    cout<<endl<<endl;
    */
    
    cout<<"---------------"<<endl<<"** Edge's information. "<<endl;
    for(auto it=m_edges.begin(); it!= m_edges.end(); it++){
        cout<<"  * id "<<it->first<<" :: ("<<(it->second).sup1<<", "<<(it->second).sup2<<" )"<<endl;
        cout<<"     edge cost is: "<<(it->second).new_bic_cost<<", "<<(it->second).new_fit_cost<<", "<<(it->second).mergecost<<endl;
        cout<<"             sup1: "<<m_supixs[(it->second).sup1].bic_cost<<", "<<m_supixs[(it->second).sup1].fit_cost<<", size: "<<m_supixs[(it->second).sup1].pixs.size()<<endl;
        cout<<"             sup2: "<<m_supixs[(it->second).sup2].bic_cost<<", "<<m_supixs[(it->second).sup2].fit_cost<<", size: "<<m_supixs[(it->second).sup2].pixs.size()<<endl;
    }
    cout<<endl;
}

void SuperPixelMerger::GetDebugImage(CDataTempl<float> &debugI, UINT32 mode){
    for(UINT32 k=0; k < m_supixs.size(); k++){
        float val = m_supixs[k].fit_cost / m_supixs[k].pixs.size();
        debugI.ResetDataFromVector(m_supixs[k].pixs, val);
    }
}

void SuperPixelMerger::Merger(){

#ifdef DEBUG_SEGMENT_MERGE_STEP2
    AssignOutputLabel();
    WriteToCSV(m_outLabelI, "./output/test_extend.csv");
#endif

    while(m_merge_seeds.size()>0){
        Seed_1D top_node(0,0.0);
        top_node = m_merge_seeds.top();
        m_merge_seeds.pop();
       
        // if the edge does not exist, continue.
        if(m_edges.find(top_node.id0)==m_edges.end()){
            continue;
        }
        // if the edge has been updated, it is a invalid one.
        if(top_node.cost != m_edges[top_node.id0].mergecost){
            continue;
        }

        if(top_node.cost > m_pParam->merge_merger_thr)
            break;
      
        if (OPEN_DEBUG)
            cout<<"Merge..."<<top_node.id0<<": "<< m_edges[top_node.id0].sup1<<", "<<m_edges[top_node.id0].sup2<<", "<<top_node.cost<<endl;
        MergeSuperPixels(m_edges[top_node.id0].sup1, m_edges[top_node.id0].sup2);
        if (OPEN_DEBUG)
            cout<<".........End. # superpixel is: "<<m_supixs.size()<< endl<<endl;

#ifdef DEBUG_SEGMENT_MERGE_STEP2
        AssignOutputLabel();
        WriteToCSV(m_outLabelI, "./output/test_shrink.csv");
        string py_command = "python pyShow.py";
        system(py_command.c_str());
#endif
    }
    //PrintOutInformation();
}

void SuperPixelMerger::UpdateSuperPixel(UINT32 sup, UINT32 edge){
    if(m_edges[edge].sup1 == 0 || m_edges[edge].sup2==0)
        return;

    m_supixs[sup].perimeter= m_edges[edge].new_perimeter; 
    m_supixs[sup].fit_cost = m_edges[edge].new_fit_cost;
    m_supixs[sup].bic_cost = m_edges[edge].new_bic_cost;
    m_supixs[sup].border   = m_edges[edge].border;
    m_pSemMat->MeanZ(m_supixs[sup].pixs, m_supixs[sup].sem_score);
}

void SuperPixelMerger::ComputeGraphWeights(){
    // compute the cost of the super pixel.
    for(auto it=m_supixs.begin(); it!= m_supixs.end(); it++){
        ComputeSuperPixelCost(it->first);
    }
    
    // compute edge's weight.
    for(auto it=m_edges.begin(); it!= m_edges.end(); it++){
        ComputeEdgeWeights(it->first);
    }
}

void SuperPixelMerger::ComputeEdgeWeights(UINT32 edge){
    auto ComputeMergeInfo = [&](DistEdge &ref_edge, bool is_row){
        DistSuperPixel &supix0 = m_supixs[ref_edge.sup1];
        DistSuperPixel &supix1 = m_supixs[ref_edge.sup2];
        UINT32 ch0 = is_row? 0 : 1;
        UINT32 ch1 = is_row? 2 : 3;
        
        // bbox.
        UINT32 bbox0_0 = supix0.border.bbox[ch0], bbox0_1 = supix0.border.bbox[ch1];
        UINT32 bbox1_0 = supix1.border.bbox[ch0], bbox1_1 = supix1.border.bbox[ch1];
        ref_edge.border.bbox[ch0] = min(bbox0_0, bbox1_0);
        ref_edge.border.bbox[ch1] = max(bbox0_1, bbox1_1);
        
        // compute cost
        vector<LineBD> &ref_linebd0 = is_row? supix0.border.border_h : supix0.border.border_v;
        vector<LineBD> &ref_linebd1 = is_row? supix1.border.border_h : supix1.border.border_v;
        vector<LineBD> &ref_edge_linebd = is_row? ref_edge.border.border_h : ref_edge.border.border_v;
        for(UINT32 k=ref_edge.border.bbox[ch0]; k<=ref_edge.border.bbox[ch1]; k++){
            UINT32 minP, maxP, size;
            if(k < bbox0_0 || k > bbox0_1){
                size = ref_linebd1[k-bbox1_0].size;
                minP = ref_linebd1[k-bbox1_0].minK;   maxP = ref_linebd1[k-bbox1_0].maxK;
            }
            else if(k < bbox1_0 || k > bbox1_1){
                size = ref_linebd0[k-bbox0_0].size;
                minP = ref_linebd0[k-bbox0_0].minK;   maxP = ref_linebd0[k-bbox0_0].maxK;
            }
            else{
                size = ref_linebd0[k-bbox0_0].size + ref_linebd1[k-bbox1_0].size;
                minP = min(ref_linebd0[k-bbox0_0].minK, ref_linebd1[k-bbox1_0].minK);   
                maxP = max(ref_linebd0[k-bbox0_0].maxK, ref_linebd1[k-bbox1_0].maxK);   
            }
            ref_edge.new_fit_cost += is_row? ComputeFitCost(k, minP, k, maxP, size) : ComputeFitCost(minP, k, maxP, k, size); 
            ref_edge.new_bic_cost += ComputeBICcost(size + 1);

            LineBD new_linebd(minP, maxP, size);
            ref_edge_linebd.push_back(new_linebd);
        }
    };
    
    // Main process.
    DistEdge &ref_edge = m_edges[edge];
    if(ref_edge.sup1 == 0 || ref_edge.sup2 ==0)
        return;
    
    // compute edge's edgeval as the mean of edgeval on border pixels.
    float edge_val = 0;
    for(SINT32 k=ref_edge.bd_pixs.size()-1; k >= 0; k--){
        UINT32 bd_k = ref_edge.bd_pixs[k];
        UINT32 py1 = m_borders[bd_k].pix1 / m_wd;
        UINT32 px1 = m_borders[bd_k].pix1 % m_wd;
        UINT32 py2 = m_borders[bd_k].pix2 / m_wd;
        UINT32 px2 = m_borders[bd_k].pix2 % m_wd;

        DpSeg dp_seg1, dp_seg2;
        m_pSegStock->GetDpSegmentByCoord(dp_seg1, py1, px1, (py1==py2? e_seg_h:e_seg_v));
        m_pSegStock->GetDpSegmentByCoord(dp_seg2, py2, px2, (py1==py2? e_seg_h:e_seg_v));
       
        UINT32 y0 = min(dp_seg1.y0, dp_seg2.y0);
        UINT32 x0 = min(dp_seg1.x0, dp_seg2.x0);
        UINT32 y1 = max(dp_seg1.y1, dp_seg2.y1);
        UINT32 x1 = max(dp_seg1.x1, dp_seg2.x1);
        m_borders[bd_k].edgeval = m_pSegStock->GetAllSegFitError(y0, x0, y1, x1);
        edge_val += m_borders[bd_k].edgeval;
    }

    ref_edge.edgeval = edge_val/(ref_edge.bd_pixs.size()+1);

    // compute the information if merge the connected two super pixels.
    ref_edge.border.ClearBorderHV();
    ref_edge.new_fit_cost = 0.0;
    ref_edge.new_bic_cost = 0.0;
    ComputeMergeInfo(ref_edge, true);
    ComputeMergeInfo(ref_edge, false);

    // compute merge cost.
    float sem_diff       = ComputeSemanticDifference(ref_edge.sup1, ref_edge.sup2);
    if((ref_edge.sup1==1 || ref_edge.sup2 == 1) &&sem_diff < m_pParam->merge_edge_semdiff_thr){
        ref_edge.mergecost = 0;
    }
    else{
        float sem_cost       = sem_diff >= m_pParam->merge_edge_semdiff_thr? m_pParam->merge_edge_semdiff_pnty : 0;
        
        DistSuperPixel &supix0 = m_supixs[ref_edge.sup1];
        DistSuperPixel &supix1 = m_supixs[ref_edge.sup2];
        ref_edge.new_perimeter = supix0.perimeter + supix1.perimeter - ref_edge.bd_pixs.size();
        float geo_cost       = ref_edge.new_perimeter/(supix0.pixs.size()-supix1.pixs.size());
        geo_cost            -= (supix0.perimeter/supix0.pixs.size() + supix1.perimeter/supix1.pixs.size());
        
        float merge_fit_cost = ref_edge.new_fit_cost - (supix0.fit_cost + supix1.fit_cost);
        float merge_bic_cost = ref_edge.new_bic_cost - (supix0.bic_cost + supix1.bic_cost);
        float fit_bic_cost   = merge_fit_cost + m_pParam->merge_edge_bic_alpha*merge_bic_cost;
        float connect_cost   = min(supix0.perimeter, supix1.perimeter)/float(ref_edge.bd_pixs.size());
        ref_edge.mergecost   = fit_bic_cost + sem_cost + m_pParam->merge_edge_geo_alpha*geo_cost + connect_cost*m_pParam->merge_edge_conn_alpha; 
    }
    // push new edge to seed stock.
    Seed_1D merge_seed(edge, ref_edge.mergecost);
    m_merge_seeds.push(merge_seed);
}

void SuperPixelMerger::ComputeSuperPixelCost(UINT32 sup){
    if(sup == 0)
        return;

    DistSuperPixel &ref_supix = m_supixs[sup];
    Border        &ref_border = ref_supix.border;

    // compute perimeter.
    for(auto it : ref_supix.adjacents){
        ref_supix.perimeter += m_edges[it.second].bd_pixs.size();
    }

    // compute semantic score.
    ref_supix.sem_score.resize(m_num_sem, 0);
    m_pSemMat->MeanZ(ref_supix.pixs, ref_supix.sem_score);
    
    // compute bounding box.
    UINT32 py, px;
    for(auto it=m_supixs[sup].pixs.begin(); it != m_supixs[sup].pixs.end(); it++){
        py = (*it) / m_wd;   px = (*it) % m_wd;
        ref_border.bbox[0] = min(py, ref_border.bbox[0]);
        ref_border.bbox[1] = min(px, ref_border.bbox[1]);
        ref_border.bbox[2] = max(py, ref_border.bbox[2]);
        ref_border.bbox[3] = max(px, ref_border.bbox[3]);
    }

    // upudate border H/V.
    UINT32 y0 = ref_border.bbox[0];
    UINT32 x0 = ref_border.bbox[1];
    ref_supix.border.ResizeBorderHV(m_ht, m_wd);
    vector<LineBD> &ref_linebd_h = ref_border.border_h;
    vector<LineBD> &ref_linebd_v = ref_border.border_v;
    for(auto it=m_supixs[sup].pixs.begin(); it != m_supixs[sup].pixs.end(); it++){
        py = (*it) / m_wd;   px = (*it) % m_wd;
        ref_linebd_h[py-y0].size += 1;
        ref_linebd_h[py-y0].minK = px < ref_linebd_h[py-y0].minK? px : ref_linebd_h[py-y0].minK; 
        ref_linebd_h[py-y0].maxK = px > ref_linebd_h[py-y0].maxK? px : ref_linebd_h[py-y0].maxK;
        ref_linebd_v[px-x0].size += 1;
        ref_linebd_v[px-x0].minK = py < ref_linebd_v[px-x0].minK? py : ref_linebd_v[px-x0].minK;
        ref_linebd_v[px-x0].maxK = py > ref_linebd_v[px-x0].maxK? py : ref_linebd_v[px-x0].maxK;
    }
    
    // compute cost based on segment fitting-err and BIC.
    UINT32 bbox_ht = ref_border.bbox[2]-y0 + 1;
    UINT32 bbox_wd = ref_border.bbox[3]-x0 + 1;
    float fit_cost = 0.0, bic_cost = 0.0;
    for(UINT32 k = 0; k < bbox_ht; k++){
        fit_cost += ComputeFitCost(k+y0, ref_linebd_h[k].minK, k+y0, ref_linebd_h[k].maxK, ref_linebd_h[k].size);
        bic_cost += ComputeBICcost(ref_linebd_h[k].size + 1);
    }
    for(UINT32 k = 0; k < bbox_wd; k++){
        fit_cost += ComputeFitCost(ref_linebd_v[k].minK, k+x0, ref_linebd_v[k].maxK, k+x0, ref_linebd_v[k].size);
        bic_cost += ComputeBICcost(ref_linebd_v[k].size + 1);
    }

    ref_supix.fit_cost = fit_cost;
    ref_supix.bic_cost = bic_cost;
}

float SuperPixelMerger::ComputeFitCost(UINT32 y0, UINT32 x0, UINT32 y1, UINT32 x1, UINT32 size){
    float fit_err  = m_pSegStock->GetAllSegFitErrorOnAny2Points(y0, x0, y1, x1);
    return (fit_err * size);
}

float SuperPixelMerger::ComputeBICcost(UINT32 numPix){
    return log(numPix + m_pParam->merge_supix_bic_addi_len);
}

float SuperPixelMerger::ComputeSemanticDifference(UINT32 sup0, UINT32 sup1){
    return _ChiDifference(m_supixs[sup0].sem_score, m_supixs[sup1].sem_score);
}


#endif
