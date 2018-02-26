#ifndef _GRAPH_HXX_
#define _GRAPH_HXX_

#include <climits>
#include "utils/DataTemplate.hxx"
using namespace std;
/*
 * Implementation is based on Fuxin's SupixelMerger, Andres's Multicut, and Guzhi's Cmodel base.
 * Class: Graph from Image. 
 * Members: borderPixel: describe border pixel pairs between super pixel.
 *          superPixel:  nodes on Graph.
 *          edgeLet: edges between superpixels.
 */

class BndryPix{
protected: 
    // Pixel indices. pix1 should on upper/left side of pix2.
    UINT32 pix1;
    UINT32 pix2;

    // super pixels they belong to
    UINT32 sup1;
    UINT32 sup2;
    
    // edge value between them.
    float edgeval;
public:
    BndryPix(UINT32 p1=0, UINT32 p2=0, UINT32 s1=0, UINT32 s2=0, float bdV=0){
        pix1 = p1 < p2? p1 : p2;
        pix2 = p1 < p2? p2 : p1; 
        sup1=s1; sup2 =s2; 
        edgeval=bdV;
    }

    void ModifySuperPixel(UINT32 new_sup, UINT32 ori_sup){
        if(sup1 == ori_sup)
            sup1 = ori_sup;
        else
            sup2 = ori_sup;
    }
};


class Edge{
protected: 
    // super pixels on edge's two side. sup1 <  sup2.
    UINT32   sup1;
    UINT32   sup2;
    
    // edge value
    float  edgeval;
    vector<UINT32> bd_pixs;
public:
    Edge(UINT32 s1=0, UINT32 s2=0, float val=0){
        sup1 = s1 < s2? s1 : s2; 
        sup2 = s1 < s2? s2 : s1; 
        edgeval=val;
    }
    
    void ModifySuperPixel(UINT32 new_sup, UINT32 ori_sup){
        if(sup1 == ori_sup)
            sup1 = ori_sup;
        else
            sup2 = ori_sup;
    }
};

// super pixel
class  Supix{
protected: 
    // bbounding box
    UINT32 bbox[4]; 

    // included pixels.
    vector<UINT32> pixs;

    // adjacent superpixes and the edge between them. <nei_supix_id, edge_id>
    map<UINT32, UINT32> adjacents;
public:
    Supix(){
        bbox[0] = UINT_MAX;
        bbox[1] = UINT_MAX;
        bbox[2] = 0;
        bbox[3] = 0;
    }
};


template<typename Node, typename Edge, typename Border>
class Graph{
protected:
    map<UINT32, Border> m_borders;
    map<UINT32, Edge>   m_edges;
    map<UINT32, Node>   m_supixs;

    const CDataTempl<UINT32>  *m_pInLabelI;
    CDataTempl<UINT32>         m_outLabelI;
    UINT32 m_ht, m_wd;

public:
    Graph(UINT32 ht, UINT32 wd){
        m_ht = ht; m_wd = wd;
        m_outLabelI.Init(ht, wd);
    }

    // Create Graph.
    void AssignInputLabel(const CDataTempl<UINT32> *pInput){
        assert(m_ht == pInput->GetYDim() && m_wd==pInput->GetXDim());
        m_pInLabelI = pInput;
    }
    void CreateGraphFromLabelI();
    
    // operators on graph
    void MergeSupixels(UINT32 sup0, UINT32 sup1);
    void MergeTwoEdge(UINT32 edge0, UINT32 edge1, UINT32 sup0, UINT32 same_sup);
    void UpdateEdge(UINT32 edge, UINT32 ori_sup, UINT32 new_sup);
    void RemoveEdge(UINT32 edge);

    // 
    void AddPixel2Supixel(UINT32 sup, UINT32 pix_idx, UINT32 py, UINT32 px);


    // empty function as API.
    void ComputeGraphWeights(){};
    void ComputeEdgeWeights(UINT32 edge){};
};

template<typename Node, typename Edge, typename Border>
void Graph<Node, Edge, Border>::AddPixel2Supixel(UINT32 sup, UINT32 pix_idx, UINT32 py, UINT32 px){
    m_supixs[sup].pixs.push_back(pix_idx);
    m_supixs[sup].bbox[0] = min(px, m_supixs[sup].bbox[0]);
    m_supixs[sup].bbox[1] = min(py, m_supixs[sup].bbox[1]);
    m_supixs[sup].bbox[2] = max(px, m_supixs[sup].bbox[2]);
    m_supixs[sup].bbox[3] = max(py, m_supixs[sup].bbox[3]);
}

template<typename Node, typename Edge, typename Border>
void Graph<Node, Edge, Border>::MergeTwoEdge(UINT32 edge0, UINT32 edge1, UINT32 sup0, UINT32 same_sup){
    for(auto it = m_edges[edge1].bd_pixs.begin(); it != m_edges[edge1].bd_pixs.end(); it++){
        m_borders[*it].ModifySuperPixel(sup0, same_sup);
        m_edges[edge0].bd_pixs.push_back(*it);
    }
    m_edges.erase();
}

// update super pixel id in edge, and the border pixels on it.
template<typename Node, typename Edge, typename Border>
void Graph<Node, Edge, Border>::UpdateEdge(UINT32 edge, UINT32 new_sup, UINT32 ori_sup){
    for(auto it = m_edges[edge].bd_pixs.begin(); it!=m_edges[edge].bd_pixs.end(); it++){
        m_borders[*it].ModifySuperPixel(new_sup, ori_sup);
    }
    m_edges[edge].ModifySuperPixel(new_sup, ori_sup);
}

// Remove edge and border pixels on it from the stock.
template<typename Node, typename Edge, typename Border>
void Graph<Node, Edge, Border>::RemoveEdge(UINT32 edge){
    for(auto it=m_edges[edge].bd_pixs.begin(); it != m_edges[edge].bd_pixs.end(); it++){
        m_borders.erase(*it); 
    }
    m_edges.erase(edge);
}

// merge super pixels sup1 into sup0.
template<typename Node, typename Edge, typename Border>
void Graph<Node, Edge, Border>::MergeSupixels(UINT32 sup0, UINT32 sup1){
    // In main process, only process case sup1 > sup0, and merge sup1 to sup0.
    if(sup0 > sup1){
        MergeSupixels(sup1, sup0);
        return;
    }

    // Main process.
    // 1. add all pixels in sup1 into sup0.
    for(auto it=m_supixs[sup1].pixs.begin(); it != m_supixs[sup1].pixs.end(); it++)
        AddPixel2Supixel(sup0, *it, *it/m_wd, *it%m_wd);

    // 2. merge sup1's adjacents to sup0's adjacents.
    for(auto it=m_supixs[sup1].adjacents.begin(); it != m_supixs[sup1].adjacents.end(); it++){
        // check if sup0 has this super pixel in its neighbour.
        if(m_supixs[sup0].adjacents.find(it->first) == m_supixs[sup0].adjacents.end()){
            m_supixs[sup0].adjacents[it->first] = it->second;
            UpdateEdge(it->second, sup0, sup1);
        }
        else{
            MergeTwoEdge(m_supixs[sup0].adjacents[it->first], sup0, it->second);
        }
        ComputeEdgeWeights(m_supixs[sup0].adjacents[it->first]);
    }

    // 3. delete sup1 and its connections with sup0.
    RemoveEdge(m_supixs[sup0].adjacents[sup1]);
    m_supixs.erase(sup1);
}


template<typename Node, typename Edge, typename Border>
void Graph<Node, Edge, Border>::CreateGraphFromLabelI(){
    auto CollectNeighbour = [&](UINT32 cur_idx, UINT32 cur_label, UINT32 &bd_cnt, UINT32 &edge_cnt, UINT32 step){
        UINT32 nei_label = m_in_labelI.GetDataByIdx(cur_idx + step);
        
        // new border pixel
        if(nei_label != cur_label){
            Border bd_pix(cur_idx, cur_idx+step, cur_label, cur_label);
            m_borders[bd_cnt] = bd_pix;
            bd_cnt += 1;

            // check if the new supix is already added into superpixel's adjacent.
            if(m_supixs[pix_label].adjacents.find(nei_label) == m_supixs[pix_label].adjacents.end()){
                Edge new_edge(pix_label, nei_label);
                m_edges[edge_cnt]  = new_edge;
                m_edges[edge_cnt].bd_pixs.push_back(bd_pix);
                
                // add the adjacent supixel to neighood.
                m_supixs[pix_label].adjacents[nei_label] = nei_cnt;
                edge_cnt += 1;
            }
            else{
                UINT32 eid = m_supixs[pix_label].adjacents[nei_label];
                m_edges[eid].bd_pixs.push_back(bd_pix);
            }
        }
    };

    // Main process
    UINT32 bd_cnt   = 0;
    UINT32 edge_cnt = 0;
    for(UINT32 py = 0; py < m_ht; py++){
        for(UINT32 px = 0; px < m_wd; px++){
            UINT32 pix_idx = py*m_wd + px;
            UINT32 pix_label = m_pInLabelI->GetDataByIdx(pix_idx);
            
            // add the pixel to super pixel.
            AddPixel2Supixel(pix_label, pix_idx, py, px);

            // look to pixel on the right and bottom side, check if exists borders.
            if(px < m_wd-1)
                CollectNeighbour(pix_idx, pix_label, bd_cnt, edge_cnt, 1);
            if(py < m_ht-1)
                CollectNeighbour(pix_idx, pix_label, bd_cnt, edge_cnt, m_wd);
        }
    } 
}


#endif
