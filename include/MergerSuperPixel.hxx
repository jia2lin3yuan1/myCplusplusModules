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

class DistEdge:public Edge{
public:
    float new_nodecost;
    float mergecost;
    
    // functions
    DistEdge():Edge(){
        new_nodecost = 0.0;
        mergecost    = 0.0;
    }
};

class DistSuperPixel:public Supix{
public:
    vector<pair<UINT32, UINT32> > m_borderH; // <minH, maxH>
    vector<pair<UINT32, UINT32> > m_borderV; // <minV, maxV>

    float nodecost;

    // functions
    DistSuperPixel():Supix(){
        nodecost = 0.0;
    }

    void ResetBorderHV(UINT32 ht, UINT32 wd){
        m_borderH.clear();
        m_borderH.resize(bbox[3]-bbox[1]+1, make_pair(wd, 0));
        m_borderV.clear();
        m_borderV.resize(bbox[2]-bbox[0]+1, make_pair(ht, 0));
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
    float ComputeLineCost(UINT32 line, UINT32 pix0, UINT32 pix1, bool is_row);

    // Merge operations
    void Merger();



};

void SuperPixelMerger::Merger(){
    while(m_merge_seeds.size()>0){
        Seed top_node(0,0,0);
        top_node = m_merge_seeds.top();
        m_merge_seeds.pop();
        
        // if the edge has been updated, update its cost, and push it back.
        if(top_node.cost != m_edges[top_node.id0].mergecost){
            top_node.cost = m_edges[top_node.id0].mergecost;
            m_merge_seeds.push(top_node);
            continue;
        }

        if(top_node.cost > m_pParam->merge_merger_thr)
            break;

        MergeSuperPixels(m_edges[top_node.id0].sup1, m_edges[top_node.id0].sup2);
    }
}

void SuperPixelMerger::UpdateSuperPixel(UINT32 sup, UINT32 edge){
    m_supixs[sup].nodecost = m_edges[edge].new_nodecost;
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
    DistEdge &ref_edge = m_edges[edge];
    
    // compute edge's edgeval as the mean of edgeval on border pixels.
    float edge_val = 0;
    for(SINT32 k=ref_edge.bd_pixs.size()-1; k >= 0; k--){
        UINT32 py1 = m_borders[k].pix1 / m_wd;
        UINT32 px1 = m_borders[k].pix1 % m_wd;
        UINT32 py2 = m_borders[k].pix2 / m_wd;
        UINT32 px2 = m_borders[k].pix2 % m_wd;

        DpSeg dp_seg1, dp_seg2;
        m_pSegStock->GetDpSegment(dp_seg1, py1, px1, (py1==py2));
        m_pSegStock->GetDpSegment(dp_seg2, py2, px2, (py1==py2));
        
        m_borders[k].edgeval = m_pSegStock->GetAllSegCost(dp_seg1.line, dp_seg1.st, dp_seg2.end, (py1==py2));
        edge_val += m_borders[k].edgeval;
    }

    ref_edge.edgeval = edge_val/(ref_edge.bd_pixs.size()+1);

    // compute the nodecost if merging the two super pixels.
    float cost = 0;
    DistSuperPixel &supix1 = m_supixs[ref_edge.sup1];
    DistSuperPixel &supix2 = m_supixs[ref_edge.sup2];
    // V direction
    UINT32 y0 = min(supix1.bbox[0], supix2.bbox[0]);
    UINT32 y1 = max(supix1.bbox[2], supix2.bbox[2]);
    for(UINT32 k=y0; k <= y1; k++){
        UINT32 minP, maxP;
        if(k < supix1.bbox[0] || k > supix1.bbox[2]){
            minP = supix2.m_borderV[k-supix2.bbox[0]].first;
            maxP = supix2.m_borderV[k-supix2.bbox[0]].second;
        }
        else if(k < supix2.bbox[0] || k > supix2.bbox[2]){
            minP = supix1.m_borderV[k-supix1.bbox[0]].first;
            maxP = supix1.m_borderV[k-supix1.bbox[0]].second;
        }
        else{
            minP = min(supix1.m_borderV[k-supix1.bbox[0]].first, supix2.m_borderV[k-supix2.bbox[0]].first);
            maxP = max(supix1.m_borderV[k-supix1.bbox[0]].second, supix2.m_borderV[k-supix2.bbox[0]].second);
        }

        cost +=  ComputeLineCost(k, minP, maxP, false);
    }
    // H direction
    UINT32 x0 = min(supix1.bbox[1], supix2.bbox[1]);
    UINT32 x1 = max(supix1.bbox[3], supix2.bbox[3]);
    for(UINT32 k=x0; k <= x1; k++){
        UINT32 minP, maxP;
        if(k < supix1.bbox[1] || k > supix1.bbox[3]){
            minP = supix2.m_borderH[k-supix2.bbox[1]].first;
            maxP = supix2.m_borderH[k-supix2.bbox[1]].second;
        }
        else if(k < supix2.bbox[1] || k > supix2.bbox[3]){
            minP = supix1.m_borderH[k-supix1.bbox[1]].first;
            maxP = supix1.m_borderH[k-supix1.bbox[1]].second;
        }
        else{
            minP = min(supix1.m_borderH[k-supix1.bbox[1]].first, supix2.m_borderH[k-supix2.bbox[1]].first);
            maxP = max(supix1.m_borderH[k-supix1.bbox[1]].second, supix2.m_borderH[k-supix2.bbox[1]].second);
        }
        cost +=  ComputeLineCost(k, minP, maxP, true);
    }
    
    ref_edge.new_nodecost = cost / (y1-y0+1 + x1-x0+1); 

    // compute merge cost.
    float old_nodecost = m_supixs[ref_edge.sup1].nodecost + m_supixs[ref_edge.sup2].nodecost;
    ref_edge.mergecost = ref_edge.new_nodecost - (old_nodecost + ref_edge.edgeval); 
}

void SuperPixelMerger::ComputeSuperPixelCost(UINT32 sup){
    // Main Process
    DistSuperPixel &ref_supix = m_supixs[sup];
    ref_supix.ResetBorderHV(m_ht, m_wd);
    UINT32 y0 = ref_supix.bbox[0];
    UINT32 x0 = ref_supix.bbox[1];

    // upudate border H/V.
    for(auto it=ref_supix.adjacents.begin(); it!=ref-supix.adjacents.end(); it ++){
        DistEdge &ref_edge = m_edges[it->second];
        for(SINT32 k=ref_edge.bd_pixs.size()-1; k>=0; k--){
            UINT32 pix = m_borders[k].GetPixelInSuperPixel(sup);
            UINT32  py = pix/m_wd, px = pix%m_wd;

            ref_supix.m_borderH[py-y0].first  = px < ref_supix.m_borderH[py-y0].first?  px : ref_supix.m_borderH[py-y0].first;       
            ref_supix.m_borderH[py-y0].second = px > ref_supix.m_borderH[py-y0].second? px : ref_supix.m_borderH[py].second;      
            ref_supix.m_borderV[px-x0].first  = py < ref_supix.m_borderV[px-x0].first?  py : ref_supix.m_borderV[px].first;       
            ref_supix.m_borderV[px-x0].second = py > ref_supix.m_borderV[px-x0].second? py : ref_supix.m_borderV[px].second;      

        }
    }

    // compute cost based on segment fitting-err and BIC.
    float cost = 0;
    for(UINT32 k = y0; k <= ref_supix.bbox[2]; k++)
        cost += ComputeLineCost(k, ref_supix.m_borderV[k-y0].first, ref_supix.m_borderV[k-y0].second, false);
    for(UINT32 k = x0; k <= ref_supix.bbox[3]; k++)
        cost += ComputeLineCost(k, ref_supix.m_borderH[k-x0].first, ref_supix.m_borderH[k-x0].second, true);

    ref_supix.nodecost = cost / (ref_supix.bbox[3]-x0+1 + ref_supix.bbox[2]-y0+1);
}

float SuperPixelMerger::ComputeLineCost(UINT32 line, UINT32 pix0, UINT32 pix1, bool is_row){
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
    fit_err        = fit_err * (pix1-pix0+1);

    float bic_cost = Log_LUT(dp_seg2.end-dp_seg1.st+1+m_pParam->merge_supix_bic_addi_len);

    return (fit_err + m_pParam->merge_supix_bic_alpha*bic_cost);
}



#endif
