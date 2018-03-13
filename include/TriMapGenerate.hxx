#ifndef _TRI_MAP_GENERATE_HXX
#define _TRI_MAP_GENERATE_HXX

#include "MergerSuperPixel.hxx"
#include "SegmentStock.hxx"

/* 
 * struct of Tri-map candidate node:
 *    sup:  super pixel ID in "super pixel stock" from super pixel merger.
 *    flag: e_tri_cand, candidate to be clustered.
 *          e_tri_seed, center of one cluster.
 *
 *    cluster_probs: probability of the node belongs to different clusters.
 */
enum TriNode_Flag {e_tri_cand=0, e_tri_seed, e_tri_crowd};

typedef struct Trimap_CandNode{
    UINT8  flag;
    float  cost; // cost to be seed
    priority_queue<Seed_1D, vector<Seed_1D>, SeedCmp_1D_Dec> cluster_probs; // <clusterId,  probablity of belonging to the cluster>
    
    Trimap_CandNode(UINT32 s=0, UINT8 f=e_tri_cand){
        flag=f;  cost = 0.0;
    }

}TriNode;

/*
 * struct of TriEdge: directed edge,
 *     edgeval: the probability of node 2 connected to node 1.
 *
 */
typedef struct Trimap_Edge{
    //UINT32 edge; // edge id in graph.
    UINT32 node1; // tri node ID.
    UINT32 node2;

    float edgeval; // probability of node2 connected to node1.
   
    Trimap_Edge(){
        //edge = 0;   
        node1 = 0; node2 = 0;
        edgeval = 0;
    }

}TriEdge;


/*
 * Class: Trimap generate based on super-pixel.
 *
 * Two things keep in mind:
 *     1. super pixel 0 means uncertained pixels, it would be labeled 0, and belongs to all instances with prob=0.5.
 *     2. super pixel 1 is the background pixels, it would be labeld 1.
 */
class Trimap_Generate{
protected:
    // input
    UINT32  m_ht,  m_wd;
    SuperPixelMerger       * m_pGraph;
    Segment_Stock          * m_pSegStock;
    CDataTempl<float>      * m_pSemMat;
    CDataTempl<float>      * m_pDistMat;

    // generated variable
    UINT32                m_numCluster;
    map<UINT32, UINT32>   m_clusters; // <nodeId, label-id>, ** label 0 is for uncertained pixels, label 1 is for background pixels.
    map<UINT32, TriNode>  m_tri_nodes; // <nodeId, triNode>
    map<Mkey_2D, TriEdge, MKey2DCmp>  m_tri_edges; // <edgeId, triEdge>
    priority_queue<Seed_1D, vector<Seed_1D>, SeedCmp_1D> m_tri_node_seeds;


    // parameters.
    const GlbParam *m_pParam;
    
public:
    Trimap_Generate(SuperPixelMerger *pMergerG, Segment_Stock *pSegStock, CDataTempl<float> *pSemMat, CDataTempl<float> *pDistMat, const GlbParam *pParam){
        m_pGraph     = pMergerG;
        m_pSegStock  = pSegStock;
        m_pSemMat    = pSemMat;
        m_pDistMat   = pDistMat;
        m_pParam     = pParam;

        m_numCluster = 1;  // default background is instance 1.
        m_ht = m_pSemMat->GetYDim();
        m_wd = m_pSemMat->GetXDim();
        
        CDataTempl<UINT32> &supixIdMap = m_pGraph->GetSuperPixelIdImage();
        InitialClusterField(supixIdMap);
    }
    // 
    void InitialClusterField(CDataTempl<UINT32> &supixIdMap);

    // Functions. 
    void GreedyGenerateTriMap();
    void GetOutputData(CDataTempl<float> &out_trimapI);

    float ComputeFitDifference(UINT32 hy0, UINT32 hx0, UINT32 hy1, UINT32 hx1, UINT32 hsup, 
                 UINT32 sk0, UINT32 sk1, UINT32 ssup, CDataTempl<UINT32> &supixIdMap);
};

float Trimap_Generate::ComputeFitDifference(UINT32 hy0, UINT32 hx0, UINT32 hy1, UINT32 hx1, UINT32 hsup,  
                                    UINT32 sk0, UINT32 sk1, UINT32 ssup, CDataTempl<UINT32> &supixIdMap){
    float fit_diff = 0;
    UINT32 fit_cnt = 0;
    if(hy0==hy1){
        UINT32 e_sup, e_x0, e_x1, seg_x0, seg_x1;
        if(hx1-hx0 > sk1-sk0 || (hx1-hx0==sk1-sk0 && hx0 < sk0)){
            e_sup = ssup;
            e_x0 = sk0;     e_x1 = sk1;
            seg_x0 = hx0;   seg_x1 = hx1;
        }
        else{
            e_sup = hsup;
            e_x0 = hx0;   e_x1 = hx1;
            seg_x0 = sk0;   seg_x1 = sk1;
        }

        // compute error
        SegFitRst *pSegInfo = &m_pSegStock->GetAllSegFitResultOnAny2Points(hy0, seg_x0, hy1, seg_x1);
        for(int x = e_x0; x <= e_x1; x ++){
            if(supixIdMap.GetData(hy0, x) != e_sup)
                continue;
            
            float hat   = (x-(int)seg_x0)*pSegInfo->w[0] + pSegInfo->b[0];
            float truth = m_pDistMat->GetData(hy0, x,  pSegInfo->ch[0]);
            fit_diff   += pow(truth - hat, 2); 
            hat         =  (x-(int)seg_x0)*pSegInfo->w[1] + pSegInfo->b[1];
            truth       = m_pDistMat->GetData(hy0, x,  pSegInfo->ch[1]);
            fit_diff   += pow(truth - hat, 2);
            fit_cnt    += 1;
        }
    }
    else{
        UINT32 e_sup, e_y0, e_y1, seg_y0, seg_y1;
        if(hy1-hy0 > sk1-sk0 || (hy1-hy0==sk1-sk0 && hy0 < sk0)){
            e_sup = ssup;
            e_y0 = sk0;     e_y1 = sk1;
            seg_y0 = hy0;   seg_y1 = hy1;
        }
        else{
            e_sup = hsup;
            e_y0 = hy0;   e_y1 = hy1;
            seg_y0 = sk0;   seg_y1 = sk1;
        }
        
        // compute error
        SegFitRst *pSegInfo = &m_pSegStock->GetAllSegFitResultOnAny2Points(seg_y0, hx0,  seg_y1, hx1);
        for(int y = e_y0; y <= e_y1; y ++){
            if(supixIdMap.GetData(y, hx0) != e_sup)
                continue;

            float hat   = (y-(int)seg_y0)*pSegInfo->w[0] + pSegInfo->b[0];
            float truth = m_pDistMat->GetData(y, hx0,   pSegInfo->ch[0]);
            fit_diff   += pow(truth - hat, 2); 
            hat         = (y-(int)seg_y0)*pSegInfo->w[1] + pSegInfo->b[1];
            truth       = m_pDistMat->GetData(y, hx0,   pSegInfo->ch[1]);
            fit_diff   += pow(truth - hat, 2); 
            fit_cnt    += 1;
        }
    }

    return fit_diff/(fit_cnt==0? 1 : fit_cnt);
}

void Trimap_Generate::InitialClusterField(CDataTempl<UINT32> &supixIdMap){
    auto MaxInVector = [](auto inVec){
        auto maxV = inVec[0];
        for(auto ele:inVec){
            maxV = ele > maxV? ele : maxV;
        }
        return maxV;
    };
    auto ComputeBelongProb = [&](UINT32 sup_host, UINT32 sup_spec){
        auto &supix_h = m_pGraph->GetSuperPixel(sup_host);
        auto &supix_s = m_pGraph->GetSuperPixel(sup_spec);
       

        // compute belong probablity via fit-err and bic-cost.
        UINT32 *p_bbox_h = supix_h.border.bbox;
        UINT32 *p_bbox_s = supix_s.border.bbox;
        UINT32 cov_y0 = max(p_bbox_h[0], p_bbox_s[0]);
        UINT32 cov_x0 = max(p_bbox_h[1], p_bbox_s[1]);
        UINT32 cov_y1 = min(p_bbox_h[2], p_bbox_s[2]);
        UINT32 cov_x1 = min(p_bbox_h[3], p_bbox_s[3]);

        // H direction
        float fit_diff_h = 0;
        vector<LineBD> &linebd_h_h = supix_h.border.border_h;
        vector<LineBD> &linebd_s_h = supix_s.border.border_h;
        for(UINT32 k = cov_y0; k <= cov_y1; k++){
            UINT32 hk     = k - p_bbox_h[0];
            UINT32 sk     = k - p_bbox_s[0];
            fit_diff_h   += ComputeFitDifference(k, linebd_h_h[hk].minK, k, linebd_h_h[hk].maxK, sup_host, 
                                              linebd_s_h[sk].minK, linebd_s_h[sk].maxK, sup_spec, supixIdMap);
        }
        fit_diff_h = fit_diff_h / (cov_y0<= cov_y1? cov_y1-cov_y0+1 : 1);
        
        // V direction.
        float fit_diff_v = 0;
        vector<LineBD> &linebd_h_v = supix_h.border.border_v;
        vector<LineBD> &linebd_s_v = supix_s.border.border_v;
        for(UINT32 k = cov_x0; k <= cov_x1; k++){
            UINT32 hk     = k - p_bbox_h[1];
            UINT32 sk     = k - p_bbox_s[1];
            fit_diff_v   += ComputeFitDifference(linebd_h_v[hk].minK, k, linebd_h_v[hk].maxK, k, sup_host, 
                                                 linebd_s_v[sk].minK, linebd_s_v[sk].maxK, sup_spec, supixIdMap);
        }
        fit_diff_v = fit_diff_v / (cov_x0<= cov_x1? cov_x1-cov_x0+1 : 1);

        float prob_h = cov_y0 < cov_y1? exp(-fit_diff_h * m_pParam->tri_edge_fit_alpha) : 0;
        float prob_v = cov_x0 < cov_x1? exp(-fit_diff_v * m_pParam->tri_edge_fit_alpha) : 0;
       
        if(cov_y0 > cov_y1)
            return prob_v;
        else if(cov_x0 > cov_x1)
            return prob_h;
        else
            return min(prob_h, prob_v);
    };
    auto ComputeSeedCost = [&](UINT32 sup){
        auto &supix = m_pGraph->GetSuperPixel(sup);
        float cost  = 0;

        // geometric cost
        float area = supix.pixs.size();
        float perim = 0;
        for(auto it:supix.adjacents){
            auto &edge = m_pGraph->GetEdge(it.second);
            perim += edge.bd_pixs.size();
        }
        float geo_cost = perim / area;
        cost += geo_cost * m_pParam->tri_seed_geo_alpha;

        // semantic cost
        float sem_prob        = MaxInVector(supix.sem_score);
        float sem_cost        = _NegativeLog(sem_prob);
        cost += sem_cost * m_pParam->tri_seed_sem_alpha;

        // fitting cost.
        cost += supix.fit_cost * m_pParam->tri_seed_fit_alpha;

        // size cost.
        cost += 1.0/log(supix.pixs.size());
        return cost;
    };

    // Main process.
    // Create TriNode and new directed edge based on super pixels.
    vector<UINT32> supIds = m_pGraph->GetAllSuperPixelsId();
    for(auto ele: supIds){
        m_tri_nodes[ele].cost = ComputeSeedCost(ele); 
        m_tri_nodes[ele].flag = e_tri_cand;

        // add the triNode as candidate seed.
        Seed_1D node(ele, m_tri_nodes[ele].cost);
        m_tri_node_seeds.push(node);

        // traverse all other supixs to create TriEdge.
        if(ele == 0)
            continue;
        
        auto &supix = m_pGraph->GetSuperPixel(ele);
        cout<<" ** cen "<<ele<<": size = "<<supix.pixs.size()<<",  bbox = ["<<supix.border.bbox[0]<<", "<<supix.border.bbox[1]<<", "<<supix.border.bbox[2]<<", "<<supix.border.bbox[3]<<" ]"<<endl;
        for(auto ele_a : supIds){
            if(ele_a == 0 || ele_a==1 || ele_a == ele)
                continue;
            // create edge.
            Mkey_2D edge_key(ele, ele_a);
            m_tri_edges[edge_key].node1    = ele;
            m_tri_edges[edge_key].node2    = ele_a;
            
            // if two super pixels have different semantic score vector.
            float sem_diff = m_pGraph->ComputeSemanticDifference(ele, ele_a);
            if(sem_diff > m_pParam->tri_edge_semdiff_thr)
                m_tri_edges[edge_key].edgeval = 0.0;
            else{
                cout<<"     nei "<< ele_a <<":: ";
                m_tri_edges[edge_key].edgeval = ComputeBelongProb(ele, ele_a);
                cout<<",   edge_prob = "<<setprecision(4)<<m_tri_edges[edge_key].edgeval<<endl;
            }
        }
    }
}

void Trimap_Generate::GreedyGenerateTriMap(){
    // label 0 are assigned to pixels that are uncertained, and could be connect to any instances.
    // label 1, is the background, it is not considered as a cluster or belong to any cluster.
    while(m_tri_node_seeds.size()>0){
        Seed_1D top_node = m_tri_node_seeds.top();
        m_tri_node_seeds.pop();
       
        // check if the node could be a cluster center.
        if(m_tri_nodes[top_node.id0].flag != e_tri_cand){
            continue;
        }
        
        // add the node to the clusters
        m_tri_nodes[top_node.id0].flag = e_tri_seed;
        if(top_node.id0 == 1){ // background class
            m_clusters[top_node.id0] = 0;
        }
        else{
            m_clusters[top_node.id0] = m_numCluster;
            m_numCluster += 1;
        }
        
        // compute the probability of other nodes in the cluster.
        for(auto it=m_tri_nodes.begin(); it!=m_tri_nodes.end(); it++){
            if(it->first == top_node.id0){
                (it->second).flag = e_tri_seed; 
                Seed_1D cls_node(m_clusters[top_node.id0], 1.01);
                (it->second).cluster_probs.push(cls_node);
            }
            else{
                Mkey_2D mkey(top_node.id0, it->first);
                Seed_1D cls_node(m_clusters[top_node.id0], m_tri_edges[mkey].edgeval);
                (it->second).cluster_probs.push(cls_node);
                
                if(m_tri_edges[mkey].edgeval > m_pParam->tri_notseed_prob_thr){
                    cout << "-- rgn "<<it->first<<" is set as crowd. with Prob "<<m_tri_edges[mkey].edgeval<<" belongs to rgn "<<top_node.id0<<endl;
                    (it->second).flag = e_tri_crowd;
                }
            }
        }
    }
}

void Trimap_Generate::GetOutputData(CDataTempl<float> &out_trimapI){
    // label 0 are assigned to pixels that are uncertained, and could be connect to any instances.
    // label 1, is the background, it is not considered as a cluster or belong to any cluster.

    
    // Assign each triNode to the closest cluster.
    out_trimapI.Init(m_ht, m_wd, m_numCluster);
    for(auto ele : m_tri_nodes){
        if(ele.first == 1){
            continue;
        }
        else if(ele.first == 0){
            auto &supix = m_pGraph->GetSuperPixel(0);
            out_trimapI.ResetDataFromVector(supix.pixs, m_pParam->tri_cluster_supix0_prob, 0);
        }
        else{
            auto &supix = m_pGraph->GetSuperPixel(ele.first);
            TriNode &tri_node = ele.second;
            while(tri_node.cluster_probs.size()>0){
                Seed_1D top_node = tri_node.cluster_probs.top();
                tri_node.cluster_probs.pop();
                
                if(top_node.id0 == 0){
                    continue;
                }
                else if(top_node.cost > m_pParam->tri_cluster_prob_thr){
                    out_trimapI.ResetDataFromVector(supix.pixs, top_node.cost, top_node.id0);
                }
                else{
                    break;
                }
            } // end of while
        } // end of if
    }// end of for ele in m_tri_nodes.
}






#endif
