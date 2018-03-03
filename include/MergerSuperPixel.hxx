#ifndef _MERGER_SUPER_PIXEL_HXX
#define _MERGER_SUPER_PIXEL_HXX

/*
 * Class: SuperPixelMerger.
 *    used to merge super pixels based on fitting error from distance map. It's also based on the Bayesian Information Criterion, which is used to balance the size and fitting error.i
 *
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
        border_h.resize(bbox[2]-bbox[0]+1, h_linebd);
        
        border_v.clear();
        LineBD v_linebd(ht, 0, 0);
        border_v.resize(bbox[3]-bbox[1]+1, v_linebd);
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
    float new_fit_cost;
    float new_bic_cost;
    
    // functions
    DistEdge(UINT32 s1=0, UINT32 s2=0, float edge=0):Edge(s1, s2, edge),border(){
        new_fit_cost = 0.0;
        new_bic_cost = 0.0;
        mergecost    = 0.0;
    }
};

class DistSuperPixel:public Supix{
public:
    Border border; 
    float fit_cost;
    float bic_cost;

    // functions
    DistSuperPixel():Supix(),border(){
        fit_cost = 0.0;
        bic_cost = 0.0;
    }
};

class SuperPixelMerger:public Graph<DistSuperPixel, DistEdge, BndryPix>{
protected:
    Segment_Stock *m_pSegStock;
    priority_queue<Seed, vector<Seed>, SeedCmp> m_merge_seeds;


    // Parameter.
    const GlbParam *m_pParam;

public:
    SuperPixelMerger(UINT32 ht, UINT32 wd, Segment_Stock *pSegStock, const GlbParam *pParam):Graph(ht, wd){
        m_pSegStock = pSegStock;
        m_pParam    = pParam;
    }
    
    // virtual function from Graph
    void UpdateSuperPixel(UINT32 sup, UINT32 edge);
    void ComputeGraphWeights();
    void ComputeEdgeWeights(UINT32 edge);
    

    // function working on adding member variables on Node and Edge.
    void ComputeSuperPixelCost(UINT32 sup);
    float ComputeFitCost(UINT32 line, UINT32 pix0, UINT32 pix1, UINT32 size, bool is_row);
    float ComputeBICcost(UINT32 numPix);

    // Merge operations
    void Merger();

    void GetDebugImage(CDataTempl<float> &debugI, UINT32 mode=0);
    void PrintOutInformation();

};

void SuperPixelMerger::PrintOutInformation(){
    cout<<"** super pixels: "<<m_supixs.size()<<endl;
    for(auto it=m_supixs.begin(); it!= m_supixs.end(); it++){
        cout<<it->first<<", ";
    }
    cout<<endl<<endl;
   
    cout<<"** Edge size: "<<m_edges.size()<<endl;
    for(auto it=m_edges.begin(); it!= m_edges.end(); it++){
        cout<<"("<<it->first<<": "<<(it->second).sup1<<", "<<(it->second).sup2<<", "<<(it->second).bd_pixs.size()<<" ), ";
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
        Seed top_node(0,0,0);
        top_node = m_merge_seeds.top();
        m_merge_seeds.pop();
       
        // if the edge does not exist, continue.
        if(m_edges.find(top_node.id0)==m_edges.end()){
            continue;
        }
        // if the edge has been updated, update its cost, and push it back.
        if(top_node.cost != m_edges[top_node.id0].mergecost){
            top_node.cost = m_edges[top_node.id0].mergecost;
            m_merge_seeds.push(top_node);
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
}

void SuperPixelMerger::UpdateSuperPixel(UINT32 sup, UINT32 edge){
    m_supixs[sup].fit_cost = m_edges[edge].new_fit_cost;
    m_supixs[sup].bic_cost = m_edges[edge].new_bic_cost;
    m_supixs[sup].border   = m_edges[edge].border;
}

void SuperPixelMerger::ComputeGraphWeights(){
    // compute the cost of the super pixel.
    for(SINT32 k = m_supixs.size()-1; k >= 0; k--){
        ComputeSuperPixelCost(k);
    }
    
    // compute edge's weight.
    for(SINT32 k = m_edges.size()-1; k >= 0; k--){
        ComputeEdgeWeights(k);

        Seed merge_seed(k, k, m_edges[k].mergecost);
        m_merge_seeds.push(merge_seed);
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
            if(ref_edge.sup1==2 && ref_edge.sup2==30 && k==318)
                int a = 0;
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
            ref_edge.new_fit_cost += ComputeFitCost(k, minP, maxP, size, is_row); 
            ref_edge.new_bic_cost += ComputeBICcost(size + 1);

            LineBD new_linebd(minP, maxP, size);
            ref_edge_linebd.push_back(new_linebd);
        }
    };
    
    // Main process.
    DistEdge &ref_edge = m_edges[edge];
    
    // compute edge's edgeval as the mean of edgeval on border pixels.
    float edge_val = 0;
    for(SINT32 k=ref_edge.bd_pixs.size()-1; k >= 0; k--){
        UINT32 bd_k = ref_edge.bd_pixs[k];
        UINT32 py1 = m_borders[bd_k].pix1 / m_wd;
        UINT32 px1 = m_borders[bd_k].pix1 % m_wd;
        UINT32 py2 = m_borders[bd_k].pix2 / m_wd;
        UINT32 px2 = m_borders[bd_k].pix2 % m_wd;

        DpSeg dp_seg1, dp_seg2;
        m_pSegStock->GetDpSegment(dp_seg1, py1, px1, (py1==py2));
        m_pSegStock->GetDpSegment(dp_seg2, py2, px2, (py1==py2));
        
        m_borders[bd_k].edgeval = m_pSegStock->GetAllSegCost(dp_seg1.line, dp_seg1.st, dp_seg2.end, (py1==py2));
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
    float merge_fit_cost = ref_edge.new_fit_cost - (m_supixs[ref_edge.sup1].fit_cost + m_supixs[ref_edge.sup2].fit_cost);
    float merge_bic_cost = ref_edge.new_bic_cost - (m_supixs[ref_edge.sup1].bic_cost + m_supixs[ref_edge.sup2].bic_cost);
    ref_edge.mergecost = merge_fit_cost + m_pParam->merge_supix_bic_alpha*merge_bic_cost;
}

void SuperPixelMerger::ComputeSuperPixelCost(UINT32 sup){
    DistSuperPixel &ref_supix = m_supixs[sup];
    Border        &ref_border = ref_supix.border; 

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
        fit_cost += ComputeFitCost(k+y0, ref_linebd_h[k].minK, ref_linebd_h[k].maxK, ref_linebd_h[k].size, true);
        bic_cost += ComputeBICcost(ref_linebd_h[k].size + 1);
    }
    for(UINT32 k = 0; k < bbox_wd; k++){
        fit_cost += ComputeFitCost(k+x0, ref_linebd_v[k].minK, ref_linebd_v[k].maxK, ref_linebd_v[k].size, false);
        bic_cost += ComputeBICcost(ref_linebd_v[k].size + 1);
    }

    ref_supix.fit_cost = fit_cost;
    ref_supix.bic_cost = bic_cost;
}

float SuperPixelMerger::ComputeFitCost(UINT32 line, UINT32 pix0, UINT32 pix1, UINT32 size, bool is_row){
    DpSeg dp_seg1, dp_seg2;
    if (is_row){
        m_pSegStock->GetDpSegment(dp_seg1, line, pix0, true);
        m_pSegStock->GetDpSegment(dp_seg2, line, pix1, true);
    }
    else{
        m_pSegStock->GetDpSegment(dp_seg1, pix0, line, false);
        m_pSegStock->GetDpSegment(dp_seg2, pix1, line, false);
    }
    float fit_err  = m_pSegStock->GetAllSegCost(line, dp_seg1.st, dp_seg2.end, is_row);
    
    return (fit_err * size);
}

float SuperPixelMerger::ComputeBICcost(UINT32 numPix){
    return Log_LUT(numPix + m_pParam->merge_supix_bic_addi_len);
}



#endif
