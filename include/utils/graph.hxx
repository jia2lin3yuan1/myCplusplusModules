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
public: 
    // Pixel indices. pix1 should on upper/left side of pix2.
    UINT32 pix1;
    UINT32 pix2;

    // super pixels they belong to
    UINT32 sup1;
    UINT32 sup2;
    
    // edge value between them.
    float edgeval;
    
    // Functions
    BndryPix(UINT32 p1=0, UINT32 p2=0, UINT32 s1=0, UINT32 s2=0, float bdV=0){
        pix1 = p1 < p2? p1 : p2;
        pix2 = p1 < p2? p2 : p1; 
        sup1=s1; sup2 =s2; 
        edgeval=bdV;
    }

    void ModifySuperPixel(UINT32 new_sup, UINT32 ori_sup){
        if(sup1 == ori_sup)
            sup1 = new_sup;
        else if(sup2==ori_sup)
            sup2 = new_sup;
        else
            cout<<"Error in border: MergeSuperPixels. ("<<sup1<<", "<<sup2<<"), ("<<ori_sup<<", "<<new_sup<<")."<<endl;
    }

    UINT32 GetPixelInSuperPixel(UINT32 sup){
        if(sup == sup1)
            return pix1;
        else
            return pix2;
    }
};


class Edge{
public: 
    // super pixels on edge's two side. sup1 <  sup2.
    UINT32   sup1;
    UINT32   sup2;
    
    // edge value
    float  edgeval;
    vector<UINT32> bd_pixs;
    
    // Functions
    Edge(UINT32 s1=0, UINT32 s2=0, float val=0){
        sup1 = s1 < s2? s1 : s2; 
        sup2 = s1 < s2? s2 : s1; 
        edgeval=val;
    }
    
    void ModifySuperPixel(UINT32 new_sup, UINT32 ori_sup){
        if(sup1 == ori_sup)
            sup1 = new_sup;
        else if(sup2 == ori_sup)
            sup2 = new_sup;
        else
            cout<<"Error in Edge: MergeSuperPixels. ("<<sup1<<", "<<sup2<<"), ("<<ori_sup<<", "<<new_sup<<")."<<endl;
    }
};

// super pixel
class  Supix{
public: 
    // included pixels.
    vector<UINT32> pixs;

    // adjacent superpixes and the edge between them. <nei_supix_id, edge_id>
    map<UINT32, UINT32> adjacents;
    
    // Functions
    Supix(){
    }
};


template<typename NODE, typename EDGE, typename BORDER>
class Graph{
protected:
    map<UINT32, BORDER> m_borders;
    map<UINT32, EDGE>   m_edges;
    map<UINT32, NODE>   m_supixs;

    const CDataTempl<UINT32>  *m_pInLabelI;
    CDataTempl<UINT32>         m_outLabelI;
    UINT32 m_ht, m_wd;

public:
    Graph(UINT32 ht, UINT32 wd){
        m_ht = ht; m_wd = wd;
        m_outLabelI.Init(ht, wd);
    }

    // Label <-> Graph.
    void AssignInputLabel(const CDataTempl<UINT32> *pInput){
        assert(m_ht == pInput->GetYDim() && m_wd==pInput->GetXDim());
        m_pInLabelI = pInput;
    }
    CDataTempl<UINT32> & AssignOutputLabel();
    void CreateGraphFromLabelI();
    
    // operators on graph
    void MergeSuperPixels(UINT32 sup0, UINT32 sup1);
   
    void MergeTwoEdge(UINT32 edge0, UINT32 edge1, UINT32 sup0, UINT32 same_sup);
    void UpdateEdge(UINT32 edge, UINT32 ori_sup, UINT32 new_sup);
    void RemoveEdge(UINT32 edge);
   
    // access the graph.
    map<UINT32, NODE> * GetSuperPixels(){return &m_supixs;} 

    // virtual function as API.
    virtual void UpdateSuperPixel(UINT32 sup, UINT32 edge){}
    virtual void ComputeGraphWeights()=0;
    virtual void ComputeEdgeWeights(UINT32 edge)=0;
};


template<typename NODE, typename EDGE, typename BORDER>
void Graph<NODE, EDGE, BORDER>::MergeTwoEdge(UINT32 edge0, UINT32 edge1, UINT32 sup0, UINT32 same_sup){
    for(auto it = m_edges[edge1].bd_pixs.begin(); it != m_edges[edge1].bd_pixs.end(); it++){
        m_borders[*it].ModifySuperPixel(sup0, same_sup);
        m_edges[edge0].bd_pixs.push_back(*it);
    }
    m_edges.erase(edge1);
}

// update super pixel id in edge, and the border pixels on it.
template<typename NODE, typename EDGE, typename BORDER>
void Graph<NODE, EDGE, BORDER>::UpdateEdge(UINT32 edge, UINT32 new_sup, UINT32 ori_sup){
    for(auto it = m_edges[edge].bd_pixs.begin(); it!=m_edges[edge].bd_pixs.end(); it++){
        m_borders[*it].ModifySuperPixel(new_sup, ori_sup);
    }
    m_edges[edge].ModifySuperPixel(new_sup, ori_sup);
}

// Remove edge and border pixels on it from the stock.
template<typename NODE, typename EDGE, typename BORDER>
void Graph<NODE, EDGE, BORDER>::RemoveEdge(UINT32 edge){
    for(auto it=m_edges[edge].bd_pixs.begin(); it != m_edges[edge].bd_pixs.end(); it++){
        m_borders.erase(*it); 
    }
    m_edges.erase(edge);
}

// merge super pixels sup1 into sup0.
template<typename NODE, typename EDGE, typename BORDER>
void Graph<NODE, EDGE, BORDER>::MergeSuperPixels(UINT32 sup0, UINT32 sup1){
    // In main process, only process case sup1 > sup0, and merge sup1 to sup0.
    if(sup0 > sup1){
        MergeSuperPixels(sup1, sup0);
        return;
    }

    // Main process.
    // 1. add all pixels in sup1 into sup0.and update sup0's attribute.
    for(auto it=m_supixs[sup1].pixs.begin(); it != m_supixs[sup1].pixs.end(); it++){
        m_supixs[sup0].pixs.push_back(*it);
    }
    UpdateSuperPixel(sup0, m_supixs[sup0].adjacents[sup1]);

    // 2. merge sup1's adjacents to sup0's adjacents.
    for(auto it=m_supixs[sup1].adjacents.begin(); it != m_supixs[sup1].adjacents.end(); it++){
        if(it->first == sup0)
            continue;
        
        // check if sup0 has this super pixel in its neighbour.
        if(m_supixs[sup0].adjacents.find(it->first) == m_supixs[sup0].adjacents.end()){
            m_supixs[sup0].adjacents[it->first] = it->second;
            UpdateEdge(it->second, sup0, sup1);
        }
        else{
            MergeTwoEdge(m_supixs[sup0].adjacents[it->first], it->second, sup0, sup1);
        }
        ComputeEdgeWeights(m_supixs[sup0].adjacents[it->first]);
    }

    // 3. update sup0's adjacents attribute if it is not in sup1's adjacents.
    for(auto it=m_supixs[sup0].adjacents.begin(); it != m_supixs[sup0].adjacents.end(); it++){
        if(it->first == sup1 || m_supixs[sup1].adjacents.find(it->first) != m_supixs[sup1].adjacents.end()){
            continue;
        }

        ComputeEdgeWeights(m_supixs[sup0].adjacents[it->first]);
    }

    // 4. delete sup1 and its connections with sup0.
    NODE &ref_supix0 = m_supixs[sup0];
    NODE &ref_supix1 = m_supixs[sup1];
    RemoveEdge(ref_supix0.adjacents[sup1]);
    for(auto it=ref_supix1.adjacents.begin(); it != ref_supix1.adjacents.end(); it++){
        NODE &ref_supix = m_supixs[it->first];
        m_supixs[it->first].adjacents.erase(sup1);
        if(it->first != sup0){
            m_supixs[it->first].adjacents[sup0] = ref_supix0.adjacents[it->first];
        }
    }
    m_supixs.erase(sup1);
}


template<typename NODE, typename EDGE, typename BORDER>
void Graph<NODE, EDGE, BORDER>::CreateGraphFromLabelI(){
    auto CollectNeighbour = [&](UINT32 cur_idx, UINT32 cur_label, UINT32 &bd_cnt, UINT32 &edge_cnt, UINT32 step){
        UINT32 nei_label = m_pInLabelI->GetDataByIdx(cur_idx + step);
        
        // new border pixel
        if(nei_label != cur_label && nei_label != 0){
            BndryPix bd_pix(cur_idx, cur_idx+step, cur_label, nei_label);
            m_borders[bd_cnt] = bd_pix;

            // check if the new supix is already added into superpixel's adjacent.
            if(m_supixs[cur_label].adjacents.find(nei_label) == m_supixs[cur_label].adjacents.end()){
                EDGE new_edge(cur_label, nei_label);
                m_edges[edge_cnt]  = new_edge;
                m_edges[edge_cnt].bd_pixs.push_back(bd_cnt);
                
                // add the adjacent supixel to neighood.
                m_supixs[cur_label].adjacents[nei_label] = edge_cnt;
                m_supixs[nei_label].adjacents[cur_label] = edge_cnt;
                edge_cnt += 1;
            }
            else{
                UINT32 eid = m_supixs[cur_label].adjacents[nei_label];
                m_edges[eid].bd_pixs.push_back(bd_cnt);
            }
            
            bd_cnt += 1;
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
            m_supixs[pix_label].pixs.push_back(pix_idx);
            if(pix_label == 0) // ignore label 0
                continue;

            // look to pixel on the right and bottom side, check if exists borders.
            if(px < m_wd-1)
                CollectNeighbour(pix_idx, pix_label, bd_cnt, edge_cnt, 1);
            if(py < m_ht-1)
                CollectNeighbour(pix_idx, pix_label, bd_cnt, edge_cnt, m_wd);
        }
    } 
}

template<typename NODE, typename EDGE, typename BORDER>
CDataTempl<UINT32> & Graph<NODE, EDGE, BORDER>::AssignOutputLabel(){
    UINT32 label = 0;
    for(auto it=m_supixs.begin(); it!= m_supixs.end(); it++){
        m_outLabelI.ResetDataFromVector((it->second).pixs, label);
        label += 1;
    }

    return m_outLabelI;
}


#endif
