#ifndef _MERGER_SUPER_PIXEL_HXX
#define _MERGER_SUPER_PIXEL_HXX

/*
 * Class: SuperPixelMerger.
 *    used to merge super pixels based on fitting error from distance map. It's also based on the Bayesian Information Criterion, which is used to balance the size and fitting error.
 *
 *   API: AssignInputLabel()
 *        CreateGraphFromLabelI()
 *        ComputeGraphProperties()
 *        Merger(num_obj)
 *
 *        GetDebugImage()
 *        AssignOutputLabel()
 */

#include "utils/graph.hxx"
#include "utils/read_write_img.hxx"

class SegEdge:public Edge{
public:
    double meanval;
    double maxval;
    double minval;

    double mrg_cost;
    double mrg_perimeter;
    double mrg_meanval;
    vector<int> mrg_hist;
    vector<int> mrg_bbox;

    SegEdge():Edge(){
        meanval = 0;
        maxval = 0;
        minval = 0;

        mrg_bbox.resize(4, 0);
    }
    
};

class SegSupix:public Supix{
public:
    double  perimeter;
    double  meanval;
    vector<int> hist;
    vector<int> bbox;

    // functions
    SegSupix():Supix(){
        perimeter = 0.0;
        meanval  = 0.0;
        bbox.resize(4, 0); // (y0, x0, y1, x1)
    }
};

class SuperPixelMerger:public Graph<SegSupix, SegEdge, BndryPix>{
protected:
    // inpput variables.
    CDataTempl<double>  *m_pPredMat;

    // generated variables
    double m_predV_hist_wd;
    int    m_predV_hist_size;
    priority_queue<Seed_1D, vector<Seed_1D>, SeedCmp_1D> m_merge_seeds;

    // Parameter.
    const GlbParam *m_pParam;

public:
    void InitialForSVM();
    SuperPixelMerger(double maxV, CDataTempl<double> *pPredMat, const GlbParam *pParam):Graph(pPredMat->GetYDim(), pPredMat->GetXDim()){
        m_pPredMat           = pPredMat;
        m_pParam             = pParam;

        m_predV_hist_wd = 0.1;
        m_predV_hist_size = int(maxV/m_predV_hist_wd) + 1;
    }
    ~SuperPixelMerger(){
    }

    // functions for unconnected graph. create a virtual edge for each two.
    void AddVirtualEdges();
    
    // virtual function from Graph
    void UpdateSuperPixel(int sup, int edge);
    void ComputeGraphProperties();
    void ComputeEdgeProperties(int edge);

    // function working on adding member variables on Node and Edge.
    void  ComputeSuperPixelProperties(int sup);
    double ComputeBICcost(double numPix);

    // Merge operations
    void Merger(int num_obj);

    // Output information
    void GetDebugImage(CDataTempl<double> &debugI, int mode=0);
    void PrintOutInformation();
};


void SuperPixelMerger::PrintOutInformation(){
    /*
    cout<<"** super pixels: "<<m_supixs.size()<<endl;
    for(auto it=m_supixs.begin(); it!= m_supixs.end(); it++){
        auto &supix = it->second;
        cout<<"*** No. "<<it->first<<",  size: " << supix.pixs.size()<<endl;
        cout << endl<<endl;
    }
    cout<<endl<<endl;
    */
    
    cout<<"---------------"<<endl<<"** Edge's information. "<<endl;
    for(auto it=m_edges.begin(); it!= m_edges.end(); it++){
        cout<<"  * id "<<it->first<<" :: ("<<(it->second).sup1<<", "<<(it->second).sup2<<" )"<<endl;
        cout<<"             sup1: size: "<<m_supixs[(it->second).sup1].pixs.size()<<endl;
        cout<<"             sup2: size: "<<m_supixs[(it->second).sup2].pixs.size()<<endl;
    }
    cout<<endl;
}

void SuperPixelMerger::GetDebugImage(CDataTempl<double> &debugI, int mode){
    for(int k=0; k < m_supixs.size(); k++){
        double val = m_supixs[k].pixs.size();
        debugI.ResetDataFromVector(m_supixs[k].pixs, val);
    }
}

/*
 * create connections between unconnected super pixels.
 */
void SuperPixelMerger::AddVirtualEdges(){
    for(auto it1 = m_supixs.begin(); it1!= m_supixs.end(); it1 ++){
        if(it1->first == 0)
            continue;

        // bubble traverse.
        for(auto it2 = it1; it2 !=m_supixs.end(); it2 ++){
            if(it2->first == 0 || it1->first) 
                continue;
            if(it1->second.adjacents.find(it2->first)!=it1->second.adjacents.end())
                continue;

            // add a new virtual edge.
            m_edges[m_edge_tot_num].Init(it1->first, it2->first);
            ComputeEdgeProperties(m_edge_tot_num);
            
            m_edge_tot_num += 1;
        }
    }
}

/*
 * Merging super pixels refering to the merging priorigy.
 *
 */
void SuperPixelMerger::Merger(int num_supix){
    // PrintOutInformation();
    while(m_merge_seeds.size()>0 && m_supixs.size() > num_supix){

        Seed_1D top_node(0,0.0);
        top_node = m_merge_seeds.top();
        m_merge_seeds.pop();
        
        // if the edge does not exist, continue.
        if(m_edges.find(top_node.id0)==m_edges.end()){
            continue;
        }
        // if the edge has been updated, it is a invalid one.
        if(top_node.cost != m_edges[top_node.id0].mrg_cost){
            continue;
        }


#ifdef DEBUG_MERGE_STEP
        GetSuperPixelIdImage();
        WriteToCSV(m_outLabelI, "./output/before_merge.csv");
#endif
        // judge by svm for whether merge or not.
        SegEdge &ref_edge = m_edges[top_node.id0];
        
        if (OPEN_DEBUG){
            cout<<"Merge..."<<top_node.id0<<": "<< ref_edge.sup1<<", "<< ref_edge.sup2<<", "<<top_node.cost<<endl;
        }
        MergeSuperPixels(ref_edge.sup1, ref_edge.sup2);
        
        if (OPEN_DEBUG)
            cout<<".........End. # superpixel is: "<<m_supixs.size()<< endl<<endl;

#ifdef DEBUG_MERGE_STEP
        GetSuperPixelIdImage();
        WriteToCSV(m_outLabelI, "./output/after_merge.csv");
        string py_command = "python pyShow.py";
        system(py_command.c_str());
#endif
    }
}

void SuperPixelMerger::UpdateSuperPixel(int sup, int edge){
    SegEdge &ref_edge = m_edges[edge];
    if(ref_edge.sup1 == 0 || ref_edge.sup2==0)
        return;

    // update properties
    SegSupix &ref_supix = m_supixs[sup];
    ref_supix.meanval   = ref_edge.mrg_meanval;
    ref_supix.perimeter = ref_edge.mrg_perimeter;
    ref_supix.hist      = ref_edge.mrg_hist;
    ref_supix.bbox      = ref_edge.mrg_bbox;
}

/*
 * compute both properties on each super pixel, and properties related to edges.
 * */
void SuperPixelMerger::ComputeGraphProperties(){
   
    // compute the cost of the super pixel.
    vector<int> rm_supixs;
    for(auto it=m_supixs.begin(); it!= m_supixs.end(); it++){
        ComputeSuperPixelProperties(it->first);
    }
    
    // compute edge's weight and merge cost for two connected super pixels.
    for(auto it=m_edges.begin(); it!= m_edges.end(); it++){
        ComputeEdgeProperties(it->first);
    }

    // compute virtual connection between un-connected super pixels.
    AddVirtualEdges();
}

void SuperPixelMerger::ComputeEdgeProperties(int edge){
    auto ComputeEdgeInfo = [&](SegEdge &ref_edge){
        ref_edge.meanval = 0;
        ref_edge.maxval = 0;
        ref_edge.minval = INT_MAX;

        double predVal0, predVal1, grad;
        for(auto it=ref_edge.bd_pixs.begin(); it != ref_edge.bd_pixs.end(); it++){
            predVal0 = m_pPredMat->GetDataByIdx(m_borders[*it].pix1);
            predVal1 = m_pPredMat->GetDataByIdx(m_borders[*it].pix2);
            grad     = predVal0<predVal1? predVal1-predVal0 : predVal0-predVal1;
            grad     = min(grad, m_pParam->merge_edge_grad_max);

            ref_edge.meanval += grad;
            if(grad < ref_edge.minval)
                ref_edge.minval = grad;
            if(grad > ref_edge.maxval)
                ref_edge.maxval = grad;
        }
        ref_edge.meanval = ref_edge.meanval/ref_edge.bd_pixs.size();
    };
    auto CostDiscount = [&](SegSupix &ref_sup){
        if(ref_sup.meanval > m_pParam->merge_supix_rm_mean_thrH && ref_sup.pixs.size() < ref_sup.perimeter*m_pParam->merge_supix_rm_a2p_thrL){
            return (double(ref_sup.pixs.size())*0.01/ref_sup.perimeter);
        }
        else{
            return double(1.0);
        }
    };

    auto ComputeMergeInfo = [&](SegEdge &ref_edge){
        SegSupix &supix0 = m_supixs[ref_edge.sup1];
        SegSupix &supix1 = m_supixs[ref_edge.sup2];

        // perimeter.
        ref_edge.mrg_perimeter = supix0.perimeter + supix1.perimeter - ref_edge.bd_pixs.size();
        
        // merge bbox.
        ref_edge.mrg_bbox[0] = min(supix0.bbox[0], supix1.bbox[0]);
        ref_edge.mrg_bbox[1] = min(supix0.bbox[1], supix1.bbox[1]);
        ref_edge.mrg_bbox[2] = max(supix0.bbox[2], supix1.bbox[2]);
        ref_edge.mrg_bbox[3] = max(supix0.bbox[3], supix1.bbox[3]);
    
        // statistical info.
        double discount_0 = CostDiscount(supix0);
        double discount_1 = CostDiscount(supix1);
        if((discount_0>=1.0 && discount_1>=1.0) || (discount_0<1.0 && discount_1<1.0)){ 
            ref_edge.mrg_meanval = (supix0.meanval*supix0.pixs.size() + supix1.meanval*supix1.pixs.size())/(supix0.pixs.size() + supix1.pixs.size());
            ref_edge.mrg_hist.resize(m_predV_hist_size, 0);
            for(int k=0; k < m_predV_hist_size; k++){
                ref_edge.mrg_hist[k] = supix0.hist[k] + supix1.hist[k];
            }
        }
        else if(discount_0 < 1.0){
            ref_edge.mrg_meanval = supix0.meanval;
            ref_edge.mrg_hist.resize(m_predV_hist_size, 0);
            for(int k=0; k < m_predV_hist_size; k++){
                ref_edge.mrg_hist[k] = supix0.hist[k];
            }
        }
        else{
            ref_edge.mrg_meanval = supix1.meanval;
            ref_edge.mrg_hist.resize(m_predV_hist_size, 0);
            for(int k=0; k < m_predV_hist_size; k++){
                ref_edge.mrg_hist[k] = supix1.hist[k];
            }
        }


        // compute merge cost.
        double supix_diff = min(m_pParam->merge_supix_diff_max, abs(supix0.meanval-supix1.meanval));
        if (ref_edge.bd_pixs.size() > 0){
            ref_edge.mrg_cost = 1.4*ref_edge.meanval + 0.6*supix_diff;
        }
        else{
            ref_edge.mrg_cost = 2 * supix_diff + m_pParam->merge_supix_unconnect_penalty;
        }
        ref_edge.mrg_cost += min(1.0, min(supix0.pixs.size(), supix1.pixs.size()) * m_pParam->merge_supix_size_alpha);

        // discount on cost
        ref_edge.mrg_cost = ref_edge.mrg_cost * min(discount_0, discount_1);
    };
    
    // Main process.
    SegEdge &ref_edge = m_edges[edge];

    // won't merge to bg (label 0)
    if(ref_edge.sup1 == 0 || ref_edge.sup2 ==0) 
        return;
    
    // update edge information.
    ComputeEdgeInfo(ref_edge);

    // compute the information if merge the connected two super pixels.
    ComputeMergeInfo(ref_edge);
  
    // push new edge to seed stock.
    if(ref_edge.mrg_cost < m_pParam->merge_merge_thrH){
        Seed_1D merge_seed(edge, ref_edge.mrg_cost);
        m_merge_seeds.push(merge_seed);
    }
}

void SuperPixelMerger::ComputeSuperPixelProperties(int sup){
    // super pixel with label 0 is the background, won't be processed.
    if(sup == 0)
        return;
    SegSupix &ref_supix = m_supixs[sup];
    
    // statistical info. from predicted matrix
    ref_supix.meanval = 0;
    ref_supix.hist.resize(m_predV_hist_size, 0);
    double predVal = 0;
    for(int k =0; k < ref_supix.pixs.size(); k ++){
        predVal = m_pPredMat->GetDataByIdx(ref_supix.pixs[k]);
        ref_supix.meanval += predVal;
        ref_supix.hist[min(m_predV_hist_size-1, int(predVal/m_predV_hist_wd))] += 1;
    }
    ref_supix.meanval /= ref_supix.pixs.size();
    
    // upudate border related information.
    int py, px;
    for(auto it : ref_supix.adjacents){
        SegEdge &ref_edge = m_edges[it.second];
        
        // compute perimeter.
        ref_supix.perimeter += ref_edge.bd_pixs.size();
        
        // compute bounding box.
        for(int k=0; k<ref_edge.bd_pixs.size(); k++){
            BndryPix &bd_pix = m_borders[ref_edge.bd_pixs[k]];
            if(sup ==bd_pix.sup1){
                py = (bd_pix.pix1)/m_wd;
                px = (bd_pix.pix1)%m_wd;
            }
            else{
                py = (bd_pix.pix2)/m_wd;
                px = (bd_pix.pix2)%m_wd;
            }
            ref_supix.bbox[0] = min(py, ref_supix.bbox[0]);
            ref_supix.bbox[1] = min(px, ref_supix.bbox[1]);
            ref_supix.bbox[2] = max(py, ref_supix.bbox[2]);
            ref_supix.bbox[3] = max(px, ref_supix.bbox[3]);
        }
    }
}

/*
 * compute Bayesian Information Cost
 */
double SuperPixelMerger::ComputeBICcost(double numPix){
    return (log(numPix + m_pParam->merge_bic_addi_len));
}


#endif
