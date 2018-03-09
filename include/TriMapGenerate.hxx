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
    
    map<UINT32, UINT32> adjacents; // <supix id in graph, TriEdgeId nei connected to current node>
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
    UINT32 edge; // edge id in graph.
    UINT32 node1; // tri node ID.
    UINT32 node2;

    float edgeval; // probability of node2 connected to node1.
   
    Trimap_Edge(){
        edge = 0;   
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
class Trimap_Gen{
protected:
    // input
    UINT32  m_ht,  m_wd;
    auto                          * m_pGraph;
    Segment_Stock                 * m_pSegStock;
    CDataTempl<float>             * m_pSemMat;

    // generated variable
    UINT32   m_numCluster;
    map<UINT32, UINT32>   m_clusters; // <nodeId, label-id>, ** label 0 is for uncertained pixels, label 1 is for background pixels.
    map<UINT32, TriNode>  m_tri_nodes; // <nodeId, triNode>
    map<UINT32, TriEdge>  m_tri_edges; // <edgeId, triEdge>
    priority_queue<Seed_1D, vector<Seed_1D>, SeedCmp_1D> m_tri_node_seeds;

    // output
    CDataTempl<float> m_out_trimapI; //<ht, wd, num_cluster+1>,

    // parameters.
    const GlbParam *m_pParam;
    
public:
    Trimap_Gen(auto *pMergerG, Segment_Stock *pSegStock, CDataTempl<float> *pSemMat, const GlbParam *pParam){
        m_pGraph     = pMergerG;
        m_pSegStock  = pSegStock;
        m_pSemMat    = pSemMat;
        m_pParam     = pParam;

        m_numCluster = 1;  // default background is instance 1.
        m_ht = m_pSemMat->GetYDim();
        m_wd = m_pSemMat->GetXDim();

        InitialClusterField();
    }
    // 
    void RecomputeGraphWeightsForTrimap();
    void InitialClusterField();

    // Functions. 
    void GreedyGenerateTriMap();
    CDataTempl<float> & GetOutputData();
};

void Trimap_Gen::RecomputeGraphWeightsForTrimap(){
    auto ComputeInvBICcost = [&m_pParam](UINT32 size){
        return 1/log(m_pParam->tri_supix_bic_scale*size + tri_supix_bic_addi_len);
    };

    // super pixels.
    vector<UINT32> supIds = m_pGraph->GetAllSuperPixelsId();
    for(auto ele: supIds){
        auto &supix = m_pGraph->GetSuperPixel(ele);
        
        float bic_cost = 0;
        // H direction, traverse Y.
        UINT32 bbox_ht = supix.border.bbox[2] - supix.border.bbox[0] + 1;
        for(UINT32 k=0; k < bbox_ht; k++){
            bic_cost += ComputeInvBICcost(supix.border.border_h[k].size);
        }
        // V direction, traverse X.
        UINT32 bbox_wd = supix.border.bbox[3] - supix.border.bbox[1] + 1;
        for(UINT32 k=0; k < bbox_wd; k++){
            bic_cost += ComputeInvBICcost(supix.border.border_v[k].size);
        }

        supix.bic_cost = bic_cost;
    }

    // edges.
    /*vector<UINT32> edgeIds = m_pGraph->GetAllEdgesId();
    for(auto ele : edgeIds){
        auto ref_edge = m_pGraph->GetEdge(ele);
    }*/
}

void Trimap_Gen::InitialClusterField(){
    auto MaxInVector = [](auto inVec){
        auto maxV = inVec[0];
        for(auto ele:inVec){
            maxV = ele > maxV? ele : maxV;
        }
    }
    auto ComputeBICcost = [&](UINT32 size){
        return log(size+m_pParam->tri_edge_bic_addi_len); 
    }
    auto ComputeBelongProb = [&](UINT32 sup_host, UINT32 sup_spec){
        auto &supix_h = m_pGraph->GetSuperPixel(sup_host);
        auto &supix_s = m_pGraph->GetSuperPixel(sup_spec);
       
        // if two super pixels have different semantic score vector.
        float sem_diff = _ChiDifference(supix_h.sem_score, supix_s.sem_score);
        if(sem_diff > m_pParam->tri_edge_semdiff_thr)
            return 0;

        // compute belong probablity via fit-err and bic-cost.
        vector<UINT32> &bbox_h = supix_h.border.bbox;
        vector<UINT32> &bbox_s = supix_s.border.bbox;
        UINT32 cov_y0 = max(bbox_h[0], bbox_s[0]);
        UINT32 cov_x0 = max(bbox_h[1], bbox_s[1]);
        UINT32 cov_y1 = min(bbox_h[2], bbox_s[2]);
        UINT32 cov_x1 = min(bbox_h[3], bbox_s[3]);

        // H direction
        float bic_cost_h = 0, fit_cost_h = 0;
        vector<LineBD> &linebd_h = supix_h.border.border_h;
        vector<LineBD> &linebd_s = supix_s.border.border_h;
        for(UINT32 k = cov_y0; k <= cov_y1; k++){
            UINT32 hk     = k - bbox_h[0];
            UINT32 sk     = k - bbox_s[0];
            UINT32 size_h = linebd_h[hk].size;
            UINT32 size_s = linebd_s[sk].size;

            float bic_conn = ComputeBICcost(size_h+size_s);
            float bic_h    = ComputeBICcost(size_h);
            float bic_s    = ComputeBICcost(size_s);
            bic_cost_h    += bic_conn - bic_h - bic_s; // to do:: this computation does not make sense.
           
            UINT32 min_x   = min(linebd_h[hk].minK, linebd_s[sk].minK);
            UINT32 max_x   = max(linebd_h[hk].maxK, linebd_s[sk].maxK);
            float fit_conn = (size_h+size_s)*m_pSegStock->GetAllSegFitErrorOnAny2Points(k, min_x, k, max_x);
            float fit_h    = size_h *m_pSegStock->GetAllSegFitErrorOnAny2Points(k, linebd_h[hk].minK, k, linebd_h[hk].maxK);
            float fit_s    = size_s *m_pSegStock->GetAllSegFitErrorOnAny2Points(k, linebd_s[sk].minK, k, linebd_s[sk].maxK);
            fit_cost_h     += fit_conn - (fit_h + fit_s);
        }
        
        // V direction.
        float bic_cost_v = 0, fit_cost_v = 0;
        linebd_h = supix_h.border.border_v;
        linebd_s = supix_s.border.border_v;
        for(UINT32 k = cov_x0; k <= cov_x1; k++){
            UINT32 hk     = k - bbox_h[1];
            UINT32 sk     = k - bbox_s[1];
            UINT32 size_h = linebd_h[hk].size;
            UINT32 size_s = linebd_s[sk].size;

            float bic_conn = ComputeBICcost(size_h+size_s);
            float bic_h    = ComputeBICcost(size_h);
            float bic_s    = ComputeBICcost(size_s);
            bic_cost_v    += bic_conn - bic_h - bic_s; // to do:: this computation does not make sense.
           
            UINT32 min_y   = min(linebd_h[hk].minK, linebd_s[sk].minK);
            UINT32 max_y   = max(linebd_h[hk].maxK, linebd_s[sk].maxK);
            float fit_conn = (size_h+size_s)*m_pSegStock->GetAllSegFitErrorOnAny2Points(min_y, k, max_x, k);
            float fit_h    = size_h *m_pSegStock->GetAllSegFitErrorOnAny2Points(linebd_h[hk].minK, k, linebd_h[hk].maxK, k);
            float fit_s    = size_s *m_pSegStock->GetAllSegFitErrorOnAny2Points(linebd_s[sk].minK, k, linebd_s[sk].maxK, k);
            fit_cost_v    += fit_conn - (fit_h + fit_s);
        }

        float cost_h = (fit_cost_h + m_pParam->tri_edge_bic_alpha*bic_cost_h) * (bbox_h[2]-bbox_h[0]+1)/(conv_y1-conv_y0+1);
        float cost_v = (fit_cost_v + m_pParam->tri_edge_bic_alpha*bic_cost_v) * (bbox_h[3]-bbox_h[1]+1)/(conv_x1-conv_x0+1);
        float cost   = sqrt(pow(cost_h, 2) + pow(cost_v, 2));
        return exp(-cost);
    };

    // Main process.
    // Create TriNode and new directed edge based on super pixels.
    UINT32 cnt_triEdge    = 0;
    vector<UINT32> supIds = m_pGraph->GetAllSuperPixelsId();
    for(auto ele: supIds){
        auto &supix = m_pGraph->GetSuperPixel(ele);
        float sem_prob        = MaxInVector(supix.sem_score);
        float sem_cost        = _NegativeLog(sem_prob);
        m_tri_nodes[ele].cost = supix.bic_cost + m_pParam->tri_seed_fit_alpha*supix.fit_cost + m_pParam->tri_seed_sem_alpha*sem_cost;
        m_tri_nodes[ele].flag = e_tri_cand;

        // add the triNode as candidate seed.
        Seed_1D node(ele, m_tri_nodes[ele].cost);
        m_tri_node_seeds.push(node);


        // traverse the adjacents to create TriEdge.
        for(auto ele_a : supix.adajacents){
            m_tri_edges[cnt_triEdge].edge    = ele_a.second;
            m_tri_edges[cnt_triEdge].node1   = ele;
            m_tri_edges[cnt_triEdge].node2   = ele_a.first;
            m_tri_edges[cnt_triEdge].edgeval = ComputeBelongProb(ele, ele_a.first);
            
            cnt_triEdge += 1;
        }
    }
}

void Trimap_Gen::GreedyGenerateTriMap(){
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
        map<UINT32, pair<float, UINT32> > cur_prob_map;
        cur_prob_map[top_node.id0]    = make_pair(1.0, 0);
        stack<UINT32>      node_s;
        node_s.push(top_node.id0);
        while(!node_s.empty()){
            UINT32 parent = node_s.top();
            node_s.pop();

            TriNode node_p = m_tri_nodes[parent];
            for(auto nei : node_p.adjacents){
                if(cur_prob_map.find(nei.first) != cur_prob_map.end()){
                    if(cur_prob_map[parent].second >= cur_prob_map[nei.first].second){
                        continue;
                    }
                    else{
                        float prob = cur_prob_map[parent].first * m_tri_edges[nei.second];
                        cur_prob_map[nei.first].first += prob;
                    }
                }
                else{
                    node_s.push(nei.first);
                    float prob = cur_prob_map[parent].first * m_tri_edges[nei.second];
                    float lvl  = cur_prob_map[parent].second + 1;
                    cur_prob_map[nei.first] = make_pair(prob. lvl);
                }
            }
        }

        // go over the probability map, 
        for(auto it = cur_prob_map.begin(); it!=cur_prob_map.end(); it++){
            TriNode tri_node = m_tri_nodes[it->first];
            if((it->second).first > m_pParam->tri_notseed_prob_thr){
                tri_node.cluster_probs.push(make_pair(m_clusters[top_node.id0], (it->second).first));
                tri_node.flag = e_tri_crowd;
            }
        } 
    }
}

CDataTempl<float> & Trimap_Gen::GetOutputData(){
    // label 0 are assigned to pixels that are uncertained, and could be connect to any instances.
    // label 1, is the background, it is not considered as a cluster or belong to any cluster.

    
    // Assign each triNode to the closest cluster.
    m_out_trimapI.Init(m_ht, m_wd, m_numCluster);
    for(auto ele : m_tri_nodes){
        if(ele.first == 1){
            continue;
        }
        else if(ele.first == 0){
            auto &supix = m_pGraph->GetSuperPixel(0);
            m_out_trimapI.ResetDataFromVector(supix.pixs, m_pParam->tri_cluster_supix0_prob, 0);
        }
        else{
            auto &supix = m_pGraph->GetSuperPixel(ele.first);
            TriNode &tri_node = ele.second;
            while(tri_node.cluster_probs.size()>0){
                Seed_1D top_node = tri_node.cluster_probs.top();
                tri_node.cluster_probs.pop();
                
                if(top_node.first == 0){
                    continue;
                }
                else if(top_node.second > m_pParam->tri_cluster_prob_thr){
                    m_out_trimapI.ResetDataFromVector(supix.pixs, tri_node.second, tri_node.first);
                }
                else{
                    break;
                }
            } // end of while
        } // end of if
    }// end of for ele in m_tri_nodes.
}




































#endif
