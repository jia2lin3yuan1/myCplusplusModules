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
#include "svm/svm.h"
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

    float      hist_w[e_dist_num_ch*HIST_W_NUM_BIN];
    UINT32     inst_id;
    string     svm_1st_str;
    string     svm_2nd_str;
    map<UINT32, UINT32> inst_count;

    // functions
    DistSuperPixel():Supix(),border(){
        fit_cost  = 0.0;
        bic_cost  = 0.0;
        perimeter = 0.0;
        memset(hist_w, 0, e_dist_num_ch*HIST_W_NUM_BIN*sizeof(float));
    }
};

class SuperPixelMerger:public Graph<DistSuperPixel, DistEdge, BndryPix>{
protected:
    // inpput variables.
    CDataTempl<UINT32> *m_pInstanceI;
    CDataTempl<float>  *m_pSemMat;
    Segment_Stock      *m_pSegStock;
   
    // for svm model
    ofstream           m_file_svm_merge;
    UINT32             m_len_svm_feature;
    UINT32             m_merge_feature_size;
    double             *m_pSVM_decVals;
    struct svm_model   *m_pSVMmodel;
    struct svm_node    *m_pSVMX;
    double *m_pMrg_maxfea, *m_pMrg_minfea;

    UINT32             m_bg_feature_size;
    struct svm_model  *m_pSVMmodel_bg;
    struct svm_node   *m_pSVMX_bg;
    double *m_pBg_maxfea, *m_pBg_minfea;

    // generated variables
    UINT32  m_num_sem;
    priority_queue<Seed_1D, vector<Seed_1D>, SeedCmp_1D> m_merge_seeds;

    // Parameter.
    const GlbParam *m_pParam;

public:
    SuperPixelMerger(CDataTempl<float> *pSemMat, Segment_Stock *pSegStock, const GlbParam *pParam, CDataTempl<UINT32> *pInstI=NULL):Graph(pSemMat->GetYDim(), pSemMat->GetXDim()){
        m_pSemMat    = pSemMat;
        m_pSegStock  = pSegStock;
        m_pParam     = pParam;
        m_pInstanceI = pInstI;
        m_num_sem = m_pSemMat->GetZDim();

        // hist_w * num_dist + bbox_size +area + fit_cost + bic_cost + num_sem_classes.
        // 2*m_len_svm_feature + bbox_size + area + (merge_fit_cost + merge_bic_cost)
        m_len_svm_feature    = e_dist_num_ch*HIST_W_NUM_BIN + 2 + 1 + 1 + 1; 
        m_merge_feature_size = 2*m_len_svm_feature + m_num_sem + 5;
        m_pSVMX        = (struct svm_node *) malloc((m_merge_feature_size)*sizeof(struct svm_node));
        m_pSVMmodel    = nullptr;
        m_pSVM_decVals = nullptr;
        m_pMrg_maxfea  = nullptr;
        m_pMrg_minfea  = nullptr;

        m_bg_feature_size = e_dist_num_ch*HIST_W_NUM_BIN + 4;
        m_pSVMX_bg     = (struct svm_node *) malloc((m_bg_feature_size)*sizeof(struct svm_node));
        m_pSVMmodel_bg = nullptr;
        m_pBg_maxfea   = nullptr;
        m_pBg_minfea   = nullptr; 

        if(m_pParam->merge_gen_svm_train_en){
            m_file_svm_merge.open("test.txt", std::ios::app);
        }

        if(m_pParam->merge_svm_en){
            m_pSVM_decVals = (double *) malloc(1*sizeof(double));
            
            string base_dir = "/home/yuanjial/Projects/Python-pure/instanceinference/code/Cython/cython_segments/include/svm/";
            string merge_path = base_dir+"12bins-new/merge_12bins.model";
            if((m_pSVMmodel=svm_load_model(merge_path.c_str()))==0){
                cout<<"Error, can't open model file."<<endl;
                exit(1);
            }
            
            string bg_path = base_dir+"12bins-new/bg_12bins.model";
            if((m_pSVMmodel_bg=svm_load_model(bg_path.c_str()))==0){
                cout<<"Error, can't open model file."<<endl;
                exit(1);
            }

            ReadClassifierFeatureNorm(base_dir+"12bins-new/merge_feature_norm.txt", base_dir+"12bins-new/bg_feature_norm.txt");
        }
    }
    ~SuperPixelMerger(){
        if(m_pParam->merge_gen_svm_train_en){
            m_file_svm_merge.close();    
        }

        if(m_pSVMmodel != nullptr){
            svm_free_and_destroy_model(&m_pSVMmodel);
        }
        if(m_pSVMX != nullptr){
            free(m_pSVMX);
        }
        if(m_pSVM_decVals != nullptr){
            free(m_pSVM_decVals);
        }
        if(m_pMrg_maxfea != nullptr){
            delete[] m_pMrg_maxfea;
        }
        if(m_pMrg_minfea != nullptr){
            delete[] m_pMrg_minfea;
        }
        
        
        if(m_pSVMmodel_bg != nullptr){
            svm_free_and_destroy_model(&m_pSVMmodel_bg);
        }
        if(m_pSVMX_bg != nullptr){
            free(m_pSVMX_bg);
        }
        if(m_pBg_maxfea != nullptr){
            delete[] m_pBg_maxfea;
        }
        if(m_pBg_minfea != nullptr){
            delete[] m_pBg_minfea;
        }
    }
    
    // virtual function from Graph
    void UpdateSuperPixel(UINT32 sup, UINT32 edge);
    void ComputeGraphWeights();
    void ComputeEdgeWeights(UINT32 edge);
    float ComputeEdgeMergeCost(DistEdge &edge, float &merge_cost);
    

    // function working on adding member variables on Node and Edge.
    void  ComputeSuperPixelCost(UINT32 sup);
    void  ComputeHistogramW(DistSuperPixel &ref_supix, UINT32 y0, UINT32 x0, UINT32 y1, UINT32 x1);
    void  NormalizeHistogramW(DistSuperPixel &ref_supis);
    float ComputeFitCost(UINT32 y0, UINT32 x0, UINT32 y1, UINT32 x1, UINT32 size);
    float ComputeBICcost(UINT32 numPix);
    float ComputeSemanticDifference(UINT32 sup0, UINT32 sup1);

    // Merge operations
    void Merger();

    // debug information
    void GetDebugImage(CDataTempl<float> &debugI, UINT32 mode=0);
    void PrintOutInformation();

    // svm classifier
    void   ReadClassifierFeatureNorm(string mrg_fname=NULL, string bg_fname=NULL);
    void   NormalizeSVMFeature(UINT32 num_node, UINT32 type=0);
    
    string SingleSuperPixelSVMFeature(UINT32 sup, UINT32 base_k, UINT32 &x_st, bool gen_str=true);
    string ConstructFinalClassifierFeature(DistEdge &edge, UINT32 &x_st, bool gen_str=true);
   
    string ConstructFinalClassifierFeature_bg(DistSuperPixel &supix, UINT32 &x_st, bool gen_str=true);
    void   WriteClassifierTrainData_bg(string fname);
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

string SuperPixelMerger::SingleSuperPixelSVMFeature(UINT32 sup, UINT32 base_k, UINT32 &x_st, bool gen_str){
    struct svm_node *pSVMX = m_pSVMX;
    UINT32 cnt = x_st;
    // bbox
    pSVMX[cnt].index = base_k;
    pSVMX[cnt].value = m_supixs[sup].border.bbox[2] - m_supixs[sup].border.bbox[0];
    cnt += 1;
    pSVMX[cnt].index = base_k+1;
    pSVMX[cnt].value = m_supixs[sup].border.bbox[3] - m_supixs[sup].border.bbox[1];
    cnt += 1;
    // rgn area.
    pSVMX[cnt].index = base_k+2;
    pSVMX[cnt].value = m_supixs[sup].pixs.size(); 
    cnt += 1;
    // fit_cost.
    pSVMX[cnt].index = base_k+3;
    pSVMX[cnt].value = m_supixs[sup].fit_cost/m_supixs[sup].pixs.size(); 
    cnt += 1;
    // bic_cost.
    pSVMX[cnt].index = base_k+4;
    pSVMX[cnt].value = m_supixs[sup].bic_cost/m_supixs[sup].pixs.size(); 
    cnt += 1;
    // histogram_w
    for(UINT32 k =0; k < e_dist_num_ch*HIST_W_NUM_BIN; k++){
        if (m_supixs[sup].hist_w[k]==0)
            continue;
        pSVMX[cnt].index = base_k+5+k;
        pSVMX[cnt].value = m_supixs[sup].hist_w[k]; 
        cnt += 1;
    }

    // construct string for output
    string str0 = "";
    if(gen_str){
        for(UINT32 k= x_st; k < cnt; k++){
            str0 += to_string(pSVMX[k].index) + ":"+ to_string(pSVMX[k].value) + " ";
        }
    }

    x_st = cnt;
    return str0;
}

void   SuperPixelMerger::ReadClassifierFeatureNorm(string mrg_fname, string bg_fname){
    int num_feature;
    string line;
    ifstream read;
    read.open(mrg_fname);
    getline(read, line);
    num_feature = stoi(line);
    assert(num_feature == m_merge_feature_size);
    m_pMrg_maxfea = new double[m_merge_feature_size];
    m_pMrg_minfea = new double[m_merge_feature_size];
    for(int k=0; k < num_feature; k++){
        getline(read, line);
        m_pMrg_minfea[k] = stod(line.substr(0, line.find(", ")));
        m_pMrg_maxfea[k] = stod(line.substr(line.find(", ")+1));
    }
    read.close();

    
    read.open(bg_fname);
    getline(read, line);
    num_feature = stoi(line);
    assert(num_feature == m_bg_feature_size);
    m_pBg_maxfea = new double[m_bg_feature_size];
    m_pBg_minfea = new double[m_bg_feature_size];
    for(int k=0; k < num_feature; k++){
        getline(read, line);
        m_pBg_minfea[k] = stod(line.substr(0, line.find(", ")));
        m_pBg_maxfea[k] = stod(line.substr(line.find(", ")+1));
    }
    read.close();
}

void SuperPixelMerger::NormalizeSVMFeature(UINT32 num_node, UINT32 type){
    struct svm_node *pSVMX = type==0? m_pSVMX : m_pSVMX_bg;
    double *pMaxfea = type==0? m_pMrg_maxfea : m_pBg_maxfea;
    double *pMinfea = type==0? m_pMrg_minfea : m_pBg_minfea;

    UINT32 idx = 0;
    for(UINT32 k =0; k < num_node; k ++){
        idx = pSVMX[k].index;
        if(pMaxfea[idx] > pMinfea[idx]){
            pSVMX[k].value = (pSVMX[k].value-pMinfea[idx])/(pMaxfea[idx]-pMinfea[idx]);
        }
    }
}

string SuperPixelMerger::ConstructFinalClassifierFeature(DistEdge &edge, UINT32 &x_st, bool gen_str){
    struct svm_node *pSVMX = m_pSVMX;
    int base_k = 2 * m_len_svm_feature;
    UINT32 area = m_supixs[edge.sup1].pixs.size()+m_supixs[edge.sup2].pixs.size();
  
    int cnt = x_st;
    pSVMX[cnt].index = base_k;
    pSVMX[cnt].value = edge.border.bbox[2]-edge.border.bbox[0];
    cnt += 1;
   
    pSVMX[cnt].index = base_k+1;
    pSVMX[cnt].value = edge.border.bbox[3]-edge.border.bbox[1];
    cnt += 1;
    
    pSVMX[cnt].index = base_k+2;
    pSVMX[cnt].value = area;
    cnt += 1;
    
    pSVMX[cnt].index = base_k+3;
    pSVMX[cnt].value = edge.new_fit_cost/area;
    cnt += 1;
    
    pSVMX[cnt].index = base_k+4;
    pSVMX[cnt].value = edge.new_bic_cost/area;
    cnt += 1;
    
    // semantic
    for(UINT32 k=0; k < m_num_sem; k++){
        float score_diff = m_supixs[edge.sup1].sem_score[k] - m_supixs[edge.sup2].sem_score[k];
        if (score_diff == 0)
            continue;
        
        pSVMX[cnt].index = base_k+5+k;
        pSVMX[cnt].value = score_diff<0? -score_diff : score_diff; 
        cnt += 1;
    }
   
    // construct string for output
    string str0 = "";
    if(gen_str){
        for(UINT32 k= x_st; k < cnt; k++){
            str0 += " " + to_string(pSVMX[k].index) + ":"+ to_string(pSVMX[k].value);
        }
    }

    x_st = cnt;
    return str0;
}

string SuperPixelMerger::ConstructFinalClassifierFeature_bg(DistSuperPixel &supix, UINT32 &x_st, bool gen_str){
    auto NeighbourMaxSize = [&](auto &sup){
        int max_semSize = 0;
        for(auto it: sup.adjacents){
            if(m_supixs[it.first].pixs.size() > max_semSize){
                max_semSize = m_supixs[it.first].pixs.size();
            }
        }
        return max_semSize;
    };
    
    UINT32 cnt = x_st;
    
    m_pSVMX_bg[cnt].index = 0;
    m_pSVMX_bg[cnt].value = supix.border.bbox[2]-supix.border.bbox[0];
    cnt += 1;
    
    m_pSVMX_bg[cnt].index = 1;
    m_pSVMX_bg[cnt].value = supix.border.bbox[3]-supix.border.bbox[1];
    cnt += 1;
    
    m_pSVMX_bg[cnt].index = 2;
    m_pSVMX_bg[cnt].value = supix.pixs.size();
    cnt += 1;
    
    // histogram_w
    for(UINT32 k =0; k < e_dist_num_ch*HIST_W_NUM_BIN; k++){
        if (supix.hist_w[k]==0)
            continue;
        m_pSVMX_bg[cnt].index = k+3;
        m_pSVMX_bg[cnt].value = supix.hist_w[k];
        cnt += 1;
    }

    // check neighbour.
    int max_neiSize = NeighbourMaxSize(supix);
    if(max_neiSize > 0){
        m_pSVMX_bg[cnt].index = e_dist_num_ch*HIST_W_NUM_BIN+3;
        m_pSVMX_bg[cnt].value = max_neiSize;
        cnt += 1;
    }
    
    
    // construct string for output
    string str0 = "";
    if(gen_str){
        for(UINT32 k= x_st; k < cnt; k++){
            str0 += " " + to_string(m_pSVMX_bg[k].index) + ":"+ to_string(m_pSVMX_bg[k].value);
        }
    }

    x_st = cnt;
    return str0;

}

void SuperPixelMerger::WriteClassifierTrainData_bg(string fname){
    auto FakeInstanceFromGroundTruth = [&](map<UINT32, UINT32> &notfake_vec){
        CDataTempl<UINT32> &seg_labelI = GetSuperPixelIdImage();
        map<UINT32, map<UINT32, UINT32> > cnt_map;
        for(UINT32 j = 0; j < m_ht; j++){
            for(UINT32 i=0; i< m_wd; i++){
                if(m_pInstanceI->GetData(j,i)==0){
                    continue; 
                }
                else if(seg_labelI.GetData(j,i)>1){
                    cnt_map[m_pInstanceI->GetData(j,i)][seg_labelI.GetData(j, i)] += 1;  
                }
                cnt_map[m_pInstanceI->GetData(j,i)][0] += 1;
            }
        }
    
        for(auto it:cnt_map){
            for(auto it2:it.second){
                if(it2.first==0){
                    continue;
                }
                else if(it2.second >= (it.second)[0]*0.6){ // IoU threshold.
                    notfake_vec[it2.first] = 1;
                }
            }
        }
    };
    
    map<UINT32, UINT32> notfake_vec;
    FakeInstanceFromGroundTruth(notfake_vec);

    string out_str;
    ofstream out_file(fname, std::ios::app); 
    for(auto it:m_supixs){
        if(it.first < 2)
            continue;
       
        UINT32 x_st = 0;
        out_str  = to_string(notfake_vec[it.first]);
        out_str += ConstructFinalClassifierFeature_bg(it.second, x_st, true);
        out_file<<out_str<<endl;
    }
    out_file.close();
}

void SuperPixelMerger::Merger(){


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


#ifdef DEBUG_SEGMENT_MERGE_STEP2
        GetSuperPixelIdImage();
        WriteToCSV(m_outLabelI, "./output/test_extend.csv");
#endif
        // judge by svm for whether merge or not.
        DistEdge &ref_edge = m_edges[top_node.id0];
        //cout<<"supixs: "<<ref_edge.sup1<<", "<<ref_edge.sup2<<":  cost:"<<top_node.cost<<endl;
        if (OPEN_DEBUG)
            cout<<"Merge..."<<top_node.id0<<": "<< ref_edge.sup1<<", "<< ref_edge.sup2<<", "<<top_node.cost<<endl;
        
        MergeSuperPixels(ref_edge.sup1, ref_edge.sup2);
        
        if (OPEN_DEBUG)
            cout<<".........End. # superpixel is: "<<m_supixs.size()<< endl<<endl;

#ifdef DEBUG_SEGMENT_MERGE_STEP2
        GetSuperPixelIdImage();
        WriteToCSV(m_outLabelI, "./output/test_shrink.csv");
        string py_command = "python pyShow.py";
        system(py_command.c_str());
#endif
    }

    // for all left super-pixels, svm judge whether they are single instances.
    if(m_pParam->merge_svm_en){
        vector<UINT32> notIns_sups;
        for(auto it:m_supixs){
            if(it.first < 2)
                continue;
            
            UINT32 x_st = 0; 
            ConstructFinalClassifierFeature_bg(it.second, x_st, false);
            m_pSVMX_bg[x_st].index = -1;

            NormalizeSVMFeature(x_st, 1);
            double pred_label = svm_predict_values(m_pSVMmodel_bg, m_pSVMX_bg, m_pSVM_decVals);
            if(pred_label == 0){
                notIns_sups.push_back(it.first);
            }
        }
        for(int k=notIns_sups.size()-1; k>=0; k --){
            DistSuperPixel &ref_sup = m_supixs[notIns_sups[k]];
            if(ref_sup.adjacents.size() == 0){
                continue;
            }
            
            float minMergeCost = 1e9;
            UINT32 neiSupix    = 1; // back ground
            for(auto it2:ref_sup.adjacents){
                if(it2.first < 2)
                    continue;
                DistEdge &ref_edge = m_edges[it2.second];
                float merge_cost;
                float sem_diff = ComputeEdgeMergeCost(ref_edge, merge_cost);
                //cout<<notIns_sups[k]<<", "<<it2.first<<" merge cost is "<<merge_cost<<endl;
                if(sem_diff < m_pParam->merge_edge_semdiff_thr){
                    merge_cost = 0;
                }
                if(merge_cost < m_pParam->merge_merger_thr*1.5){
                    if(minMergeCost > merge_cost){
                        minMergeCost = merge_cost;
                        neiSupix     = it2.first;
                    }
                }
            }
            MergeSuperPixels(neiSupix, notIns_sups[k]);
        }
    }
    
    //PrintOutInformation();
    if(m_pParam->merge_gen_svm_train_en){
        WriteClassifierTrainData_bg("test_bg.txt");
    }
}

void SuperPixelMerger::UpdateSuperPixel(UINT32 sup, UINT32 edge){
    if(m_edges[edge].sup1 == 0 || m_edges[edge].sup2==0)
        return;

    DistSuperPixel &ref_supix = m_supixs[sup];
    ref_supix.perimeter= m_edges[edge].new_perimeter; 
    ref_supix.fit_cost = m_edges[edge].new_fit_cost;
    ref_supix.bic_cost = m_edges[edge].new_bic_cost;
    ref_supix.border   = m_edges[edge].border;

    DistSuperPixel &mrg_supix = sup==m_edges[edge].sup1? m_supixs[m_edges[edge].sup2] : m_supixs[m_edges[edge].sup1];
    // sem_score;
    for(UINT32 k = 0; k < m_num_sem; k ++){
        ref_supix.sem_score[k] = (ref_supix.sem_score[k]*ref_supix.pixs.size() + mrg_supix.sem_score[k]*mrg_supix.pixs.size())/(ref_supix.pixs.size()+mrg_supix.pixs.size());
    }

    // instance count
    if(m_pParam->merge_gen_svm_train_en){
        UINT32 maxV = ref_supix.inst_count[ref_supix.inst_id];
        for(auto it: mrg_supix.inst_count){
            ref_supix.inst_count[it.first] += it.second;
            if(ref_supix.inst_count[it.first] > maxV){
                ref_supix.inst_id = it.first;
            }
        }

        UINT32 x_st = 0;
        ref_supix.svm_1st_str = SingleSuperPixelSVMFeature(sup, 0, x_st, true);
        ref_supix.svm_2nd_str = SingleSuperPixelSVMFeature(sup, m_len_svm_feature, x_st, true);
    }
    
    // histogram_w
    Border   &ref_border = ref_supix.border;
    UINT32 y0            = ref_border.bbox[0];
    UINT32 x0            = ref_border.bbox[1];
    vector<LineBD> &ref_linebd_h = ref_border.border_h;
    vector<LineBD> &ref_linebd_v = ref_border.border_v;
    UINT32 bbox_ht = ref_border.bbox[2]-y0 + 1;
    UINT32 bbox_wd = ref_border.bbox[3]-x0 + 1;
    for(UINT32 k = 0; k < bbox_ht; k++){
        ComputeHistogramW(ref_supix, k+y0, ref_linebd_h[k].minK, k+y0, ref_linebd_h[k].maxK);
    }
    for(UINT32 k = 0; k < bbox_wd; k++){
        ComputeHistogramW(ref_supix, ref_linebd_v[k].minK, k+x0, ref_linebd_v[k].maxK, k+x0);
    }
    NormalizeHistogramW(ref_supix);
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

float SuperPixelMerger::ComputeEdgeMergeCost(DistEdge &ref_edge, float &merge_cost){
    float sem_diff       = ComputeSemanticDifference(ref_edge.sup1, ref_edge.sup2);
    float sem_cost       = sem_diff >= m_pParam->merge_edge_semdiff_thr? m_pParam->merge_edge_semdiff_pnty : 0;
    
    DistSuperPixel &supix0 = m_supixs[ref_edge.sup1];
    DistSuperPixel &supix1 = m_supixs[ref_edge.sup2];
    ref_edge.new_perimeter = supix0.perimeter + supix1.perimeter - ref_edge.bd_pixs.size();
    float geo_cost         = ref_edge.new_perimeter/(supix0.pixs.size()-supix1.pixs.size());
    geo_cost              -= (supix0.perimeter/supix0.pixs.size() + supix1.perimeter/supix1.pixs.size());
    
    float merge_fit_cost = ref_edge.new_fit_cost - (supix0.fit_cost + supix1.fit_cost);
    float merge_bic_cost = ref_edge.new_bic_cost - (supix0.bic_cost + supix1.bic_cost);
    float fit_bic_cost   = merge_fit_cost + m_pParam->merge_edge_bic_alpha*merge_bic_cost;
    float connect_cost   = min(supix0.perimeter, supix1.perimeter)/float(ref_edge.bd_pixs.size());
    merge_cost           = fit_bic_cost + sem_cost + m_pParam->merge_edge_geo_alpha*geo_cost + connect_cost*m_pParam->merge_edge_conn_alpha;

    return sem_diff;
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

    // compute the information if merge the connected two super pixels.
    ref_edge.border.ClearBorderHV();
    ref_edge.new_fit_cost = 0.0;
    ref_edge.new_bic_cost = 0.0;
    ComputeMergeInfo(ref_edge, true);
    ComputeMergeInfo(ref_edge, false);

    // compute merge cost.
    float sem_diff = ComputeEdgeMergeCost(ref_edge, ref_edge.mergecost);
    if((ref_edge.sup1==1 || ref_edge.sup2 == 1) &&sem_diff < m_pParam->merge_edge_semdiff_thr){
        ref_edge.mergecost = 0;
    }
    
    // push new edge to seed stock.
    if(ref_edge.mergecost < m_pParam->merge_merger_thr){
        if(m_pParam->merge_svm_en){
            UINT32 x_st = 0;
            SingleSuperPixelSVMFeature(ref_edge.sup1, 0, x_st, false);
            SingleSuperPixelSVMFeature(ref_edge.sup2, m_len_svm_feature, x_st, false);
            ConstructFinalClassifierFeature(ref_edge, x_st, false);
            m_pSVMX[x_st].index = -1;
            
            NormalizeSVMFeature(x_st, 0);
            double pred_label = svm_predict_values(m_pSVMmodel, m_pSVMX, m_pSVM_decVals);
            float merge_cost   = (float)(m_pSVM_decVals[0]);
            if((pred_label == 1 && merge_cost < 0) || (pred_label==0 && merge_cost > 0)){
                merge_cost = -1 * merge_cost; 
            }
            if(merge_cost > m_pParam->merge_svm_dec_thr){
                Seed_1D merge_seed(edge, ref_edge.mergecost);
                m_merge_seeds.push(merge_seed);
            }
        }
        else{
            Seed_1D merge_seed(edge, ref_edge.mergecost);
            m_merge_seeds.push(merge_seed);
        }
    }

    // write edge related info for svm training data.
    if(m_pParam->merge_gen_svm_train_en){
        UINT32 x_st = 0;
        string cls_str = (m_supixs[ref_edge.sup1].inst_id==m_supixs[ref_edge.sup2].inst_id? to_string(1) : to_string(0));
        string mrg_str = ConstructFinalClassifierFeature(ref_edge, x_st, true);  
        
        string out_str1 = cls_str + (" " + m_supixs[ref_edge.sup1].svm_1st_str + m_supixs[ref_edge.sup2].svm_2nd_str) + mrg_str;
        m_file_svm_merge<<out_str1<<endl;
        
        // string out_str2 = cls_str + (" " + m_supixs[ref_edge.sup2].svm_1st_str + m_supixs[ref_edge.sup1].svm_2nd_str) + mrg_str;
        // m_file_svm_merge<<out_str2<<endl;
    }
}

void SuperPixelMerger::ComputeSuperPixelCost(UINT32 sup){
    if(sup == 0)
        return;

    DistSuperPixel &ref_supix = m_supixs[sup];
    Border        &ref_border = ref_supix.border;

    // compute bounding box.
    UINT32 py, px;
    for(auto it=ref_supix.pixs.begin(); it != ref_supix.pixs.end(); it++){
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
    for(auto it=ref_supix.pixs.begin(); it != ref_supix.pixs.end(); it++){
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
        ComputeHistogramW(ref_supix, k+y0, ref_linebd_h[k].minK, k+y0, ref_linebd_h[k].maxK);
    }
    for(UINT32 k = 0; k < bbox_wd; k++){
        fit_cost += ComputeFitCost(ref_linebd_v[k].minK, k+x0, ref_linebd_v[k].maxK, k+x0, ref_linebd_v[k].size);
        bic_cost += ComputeBICcost(ref_linebd_v[k].size + 1);
        ComputeHistogramW(ref_supix, ref_linebd_v[k].minK, k+x0, ref_linebd_v[k].maxK, k+x0);
    }

    NormalizeHistogramW(ref_supix);
    ref_supix.fit_cost = fit_cost;
    ref_supix.bic_cost = bic_cost;
    
    // compute instance information if needed
    if(m_pParam->merge_gen_svm_train_en){
        for(UINT32 k =0; k < ref_supix.pixs.size(); k ++){
            ref_supix.inst_count[m_pInstanceI->GetDataByIdx(ref_supix.pixs[k])] += 1;
        }

        UINT32 max_cnt    = 0;
        ref_supix.inst_id = 0;
        for(auto it:ref_supix.inst_count){
            if(it.second > max_cnt){
                ref_supix.inst_id = it.first;
                max_cnt           = it.second;
            }
        }
        UINT32 x_st = 0; 
        ref_supix.svm_1st_str = SingleSuperPixelSVMFeature(sup, 0, x_st, true);
        ref_supix.svm_2nd_str = SingleSuperPixelSVMFeature(sup, m_len_svm_feature, x_st, true);
    }

    // compute perimeter.
    for(auto it : ref_supix.adjacents){
        ref_supix.perimeter += m_edges[it.second].bd_pixs.size();
    }

    // compute semantic score.
    ref_supix.sem_score.resize(m_num_sem, 0);
    m_pSemMat->MeanZ(ref_supix.pixs, ref_supix.sem_score);
    
}

void SuperPixelMerger::ComputeHistogramW(DistSuperPixel &ref_supix, UINT32 y0, UINT32 x0, UINT32 y1, UINT32 x1){
    SegFitRst &segInfo = m_pSegStock->GetAllSegFitResultOnAny2Points(y0, x0, y1, x1);
    UINT32 bin_w0 = vote2histogram_w(segInfo.w[0]);
    UINT32 bin_w1 = vote2histogram_w(segInfo.w[1]);
    if(y0 == y1){
        ref_supix.hist_w[e_dist_lft*HIST_W_NUM_BIN+bin_w0] += 1;
        ref_supix.hist_w[e_dist_rht*HIST_W_NUM_BIN+bin_w1] += 1;
    }
    else{
        ref_supix.hist_w[e_dist_bot*HIST_W_NUM_BIN+bin_w0] += 1;
        ref_supix.hist_w[e_dist_top*HIST_W_NUM_BIN+bin_w1] += 1;
    }
}
void  SuperPixelMerger::NormalizeHistogramW(DistSuperPixel &ref_supix){
    for(UINT32 k=0; k < e_dist_num_ch; k ++){
        float sum = 0;
        for(UINT32 j=0; j < HIST_W_NUM_BIN; j++){
           sum += ref_supix.hist_w[k*HIST_W_NUM_BIN+j];
        }
        sum = sum<1? 1 : sum;
        for(UINT32 j=0; j < HIST_W_NUM_BIN; j++){
           ref_supix.hist_w[k*HIST_W_NUM_BIN+j] = ref_supix.hist_w[k*HIST_W_NUM_BIN+j]/sum;
        }
    }
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
