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

typedef struct BorderSeg{
    UINT32 x_h; // in H, x coord of the border pix on another side of the areal
    UINT32 y_v; // in V, y coord of the border pix on another side of the areal
    
    float  bias[e_dist_num_ch]; // segment fitting information for the closed H/V line.
    float  weight[e_dist_num_ch];
    float  fit_err[e_fit_num];

}BdSeg;

typedef struct BorderInfo{
    // bounding box, <y0, x0, y1, x1>.
    UINT32 bbox[4];
    map<Mkey_3D, BdSeg, MKey3DCmp> border_seg;
    
    BorderInfo(){
        Reset();
    }

    void Reset(){
        bbox[0] = UINT_MAX;
        bbox[1] = UINT_MAX;
        bbox[2] = 0;
        bbox[3] = 0;

        border_seg.clear();
    }

    void MergeBoundingBox(UINT32 *bbox1, UINT32 *bbox2){
        bbox[0] = min(bbox1[0], bbox2[0]);
        bbox[1] = min(bbox1[1], bbox2[1]);
        bbox[2] = max(bbox1[2], bbox2[2]);
        bbox[3] = max(bbox1[3], bbox2[3]);
    }

    void RemoveOneBorderSeg(Mkey_3D key){
        border_seg.erase(key);
    }

    void AssignOneBorderSeg(Mkey_3D key, UINT32 oppos_xy, SegFitRst &seg_fit, bool is_row){
        if(is_row){
            border_seg[key].x_h        = oppos_xy;
            border_seg[key].bias[0]    = seg_fit.b[0];
            border_seg[key].bias[1]    = seg_fit.b[1];
            border_seg[key].weight[0]  = seg_fit.w[0];
            border_seg[key].weight[1]  = seg_fit.w[1];
            border_seg[key].fit_err[0] = seg_fit.fit_err;
        }
        else{
            border_seg[key].y_v        = oppos_xy;
            border_seg[key].bias[2]    = seg_fit.b[0];
            border_seg[key].bias[3]    = seg_fit.b[1];
            border_seg[key].weight[2]  = seg_fit.w[0];
            border_seg[key].weight[3]  = seg_fit.w[1];
            border_seg[key].fit_err[1] = seg_fit.fit_err;
        }
    }

}Border;

class DistEdge:public Edge{
public:
    Border border;

    float mergecost;
    float new_fit_cost;
    float new_bic_cost;
    float new_perimeter;
    float new_bias[e_dist_num_ch];
    
    // functions
    DistEdge(UINT32 s1=0, UINT32 s2=0, float edge=0):Edge(s1, s2, edge),border(){
        Reset();
    }

    void Reset(){
        new_fit_cost = 0.0;
        new_bic_cost = 0.0;
        mergecost    = 0.0;
        new_perimeter= 0.0;
        memset(new_bias, 0, e_dist_num_ch*sizeof(float));

        border.Reset();
    }
};

class DistSuperPixel:public Supix{
public:
    vector<float> sem_score;
    Border border;
    float  sum_bias[e_dist_num_ch];
    float  sum_fit_cost;
    float  sum_bic_cost;
    float  perimeter;

    float      hist_w[e_dist_num_ch*HIST_W_NUM_BIN];
    UINT32     inst_id;
    string     svm_1st_str;
    string     svm_2nd_str;
    map<UINT32, UINT32> inst_count;

    // functions
    DistSuperPixel():Supix(),border(){
        sum_fit_cost  = 0.0;
        sum_bic_cost  = 0.0;
        perimeter = 0.0;
        memset(sum_bias, 0, e_dist_num_ch*sizeof(float));
        
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

    // function working on adding member variables on Node and Edge.
    void  ComputeSuperPixelCost(UINT32 sup);
    void  ComputeHistogramW(DistSuperPixel &ref_supix);
    void  NormalizeHistogramW(DistSuperPixel &ref_supis);
    float ComputeFitCost(BdSeg &ref_bdSeg, float py, float px, bool is_row);
    float ComputeBICcost(float numPix);
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
        cout<<"             sup1: "<<m_supixs[(it->second).sup1].sum_bic_cost<<", "<<m_supixs[(it->second).sup1].sum_fit_cost<<", size: "<<m_supixs[(it->second).sup1].pixs.size()<<endl;
        cout<<"             sup2: "<<m_supixs[(it->second).sup2].sum_bic_cost<<", "<<m_supixs[(it->second).sup2].sum_fit_cost<<", size: "<<m_supixs[(it->second).sup2].pixs.size()<<endl;
    }
    cout<<endl;
}

void SuperPixelMerger::GetDebugImage(CDataTempl<float> &debugI, UINT32 mode){
    for(UINT32 k=0; k < m_supixs.size(); k++){
        float val = m_supixs[k].sum_fit_cost / m_supixs[k].pixs.size();
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
    pSVMX[cnt].value = m_supixs[sup].sum_fit_cost/m_supixs[sup].pixs.size(); 
    cnt += 1;
    // bic_cost.
    pSVMX[cnt].index = base_k+4;
    pSVMX[cnt].value = m_supixs[sup].sum_bic_cost/m_supixs[sup].pixs.size(); 
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
                else if(it2.second >= ((it.second)[0] + m_supixs[it2.first].pixs.size()-it2.second)*0.4){ // IoU threshold.
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


    PrintOutInformation();
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
        if (OPEN_DEBUG){
            cout<<"Merge..."<<top_node.id0<<": "<< ref_edge.sup1<<", "<< ref_edge.sup2<<", "<<top_node.cost<<endl;
        }
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
                if(it2.first < 2) // background.
                    continue;
                if(minMergeCost > m_edges[it2.second].mergecost){
                    minMergeCost = m_edges[it2.second].mergecost;
                    neiSupix     = it2.first;
                }
            }
            if(minMergeCost < m_pParam->merge_merger_thr*1.5){
                MergeSuperPixels(neiSupix, notIns_sups[k]);
            }
        }
    }
    
    if(m_pParam->merge_gen_svm_train_en){
        WriteClassifierTrainData_bg("test_bg.txt");
    }
}

void SuperPixelMerger::UpdateSuperPixel(UINT32 sup, UINT32 edge){
    DistEdge &ref_edge = m_edges[edge];
    if(ref_edge.sup1 == 0 || ref_edge.sup2==0)
        return;

    DistSuperPixel &ref_supix = m_supixs[sup];
    ref_supix.perimeter    = ref_edge.new_perimeter; 
    ref_supix.sum_fit_cost = ref_edge.new_fit_cost;
    ref_supix.sum_bic_cost = ref_edge.new_bic_cost;
    ref_supix.border       = ref_edge.border;
    memcpy(ref_supix.sum_bias, ref_edge.new_bias, e_dist_num_ch*sizeof(float));

    // sem_score;
    DistSuperPixel &mrg_supix = sup==ref_edge.sup1? m_supixs[ref_edge.sup2] : m_supixs[ref_edge.sup1];
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
    ComputeHistogramW(ref_supix);
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

void SuperPixelMerger::ComputeEdgeWeights(UINT32 edge){
    auto ConstructNewRegionInfo = [&](DistEdge &ref_edge){
        DistSuperPixel &ref_ed_sup0 = m_supixs[ref_edge.sup1];
        DistSuperPixel &ref_ed_sup1 = m_supixs[ref_edge.sup2];
        
        // merge border in 2 superpixel into edge's border.
        for(auto it: ref_ed_sup0.border.border_seg){
            ref_edge.border.border_seg[it.first] = it.second;
        }
        for(auto it: ref_ed_sup1.border.border_seg){
            ref_edge.border.border_seg[it.first] = it.second;
        }

        // statistical information.
        ref_edge.new_perimeter = ref_ed_sup0.perimeter + ref_ed_sup1.perimeter - 2*ref_edge.bd_pixs.size();
        ref_edge.new_fit_cost  = ref_ed_sup0.sum_fit_cost + ref_ed_sup1.sum_fit_cost;
        ref_edge.new_bic_cost  = ref_ed_sup0.sum_fit_cost + ref_ed_sup1.sum_fit_cost;
        for(UINT32 k=0; k < e_dist_num_ch; k ++){
            ref_edge.new_bias[k] = ref_ed_sup0.sum_bias[k] + ref_ed_sup1.sum_bias[k];
        }

        // update border along the connected edge. Also compute the merge related cost.
        for(UINT32 k=0; k < ref_edge.bd_pixs.size(); k ++){
            // get border pixel information.
            BndryPix &bd_pix = m_borders[ref_edge.bd_pixs[k]];
            DistSuperPixel &ref_bd_sup0 = m_supixs[bd_pix.sup1];
            DistSuperPixel &ref_bd_sup1 = m_supixs[bd_pix.sup2];

            UINT32 py0 = (bd_pix.pix1) / m_wd;   
            UINT32 px0 = (bd_pix.pix1) % m_wd;
            UINT32 py1 = (bd_pix.pix2) / m_wd;   
            UINT32 px1 = (bd_pix.pix2) % m_wd;
            
            
            if(py0 == py1){
                UINT32 is_h_line = 1;
                Mkey_3D pt0_key(py0, px0, is_h_line);
                Mkey_3D pt1_key(py1, px1, is_h_line);
                
                // find segment of the merged segment. Remove cost of small segment in each region.
                UINT32 mrg_px0, mrg_px1;
                mrg_px0 = ref_bd_sup0.border.border_seg[pt0_key].x_h; 
                mrg_px1 = ref_bd_sup1.border.border_seg[pt1_key].x_h;
                assert(mrg_px0 <= mrg_px1);
            
                // get segment fitting info for the merged segment.
                SegFitRst seg_fit = m_pSegStock->GetAllSegFitResultOnAny2Points(py0, mrg_px0, py1, mrg_px1);
            
                // remove old border_pixels on the edge.
                ref_edge.border.RemoveOneBorderSeg(pt0_key);
                
                // update border, compute related cost.
                Mkey_3D mrg_lft_key(py0, mrg_px0, is_h_line);
                ref_edge.border.AssignOneBorderSeg(mrg_lft_key, mrg_px1, seg_fit, true);
                Mkey_3D mrg_rht_key(py0, mrg_px1, is_h_line);
                ref_edge.border.AssignOneBorderSeg(mrg_rht_key, mrg_px0, seg_fit, true);

                ref_edge.new_fit_cost -= ComputeFitCost(ref_bd_sup0.border.border_seg[pt0_key], py0, px0, true);
                ref_edge.new_fit_cost -= ComputeFitCost(ref_bd_sup1.border.border_seg[pt1_key], py1, px1, true);
                ref_edge.new_fit_cost += ComputeFitCost(ref_edge.border.border_seg[mrg_lft_key], py0, mrg_px0, true);

                ref_edge.new_bic_cost -= ComputeBICcost(abs(static_cast<float>(px0)-mrg_px0+1.0));
                ref_edge.new_bic_cost -= ComputeBICcost(abs(static_cast<float>(mrg_px1)-px1+1.0));
                ref_edge.new_bic_cost += ComputeBICcost(abs(static_cast<float>(mrg_px1)-mrg_px0+1.0));

                for(UINT32 kk = 0; kk < 2; kk ++){
                    ref_edge.new_bias[kk] -= ref_bd_sup0.border.border_seg[pt0_key].bias[kk];
                    ref_edge.new_bias[kk] -= ref_bd_sup1.border.border_seg[pt1_key].bias[kk];
                    ref_edge.new_bias[kk] += ref_edge.border.border_seg[mrg_lft_key].bias[kk];
                }
            }
            else{
                UINT32 is_h_line = 0;
                Mkey_3D pt0_key(py0, px0, is_h_line);
                Mkey_3D pt1_key(py1, px1, is_h_line);
            
                // find segment of the merged segment. Remove cost of small segment in each region.
                UINT32 mrg_py0, mrg_py1;
                mrg_py0 = ref_bd_sup0.border.border_seg[pt0_key].y_v; 
                mrg_py1 = ref_bd_sup1.border.border_seg[pt1_key].y_v;
                assert(mrg_py0 <= mrg_py1);

                // get segment fitting info for the merged segment.
                SegFitRst seg_fit = m_pSegStock->GetAllSegFitResultOnAny2Points(mrg_py0, px0, mrg_py1, px1);

                // remove old border_pixels on the edge.
                ref_edge.border.RemoveOneBorderSeg(pt1_key);
                
                // update border, compute related cost.
                Mkey_3D mrg_top_key(mrg_py0, px0, is_h_line);
                ref_edge.border.AssignOneBorderSeg(mrg_top_key, mrg_py1, seg_fit, false);
                Mkey_3D mrg_bot_key(mrg_py1, px0, is_h_line);
                ref_edge.border.AssignOneBorderSeg(mrg_bot_key, mrg_py0, seg_fit, false);
                
                ref_edge.new_fit_cost -= ComputeFitCost(ref_bd_sup0.border.border_seg[pt0_key], py0, px0, false);
                ref_edge.new_fit_cost -= ComputeFitCost(ref_bd_sup1.border.border_seg[pt1_key], py1, px1, false);
                ref_edge.new_fit_cost += ComputeFitCost(ref_edge.border.border_seg[mrg_top_key], mrg_py0, px0, false);

                ref_edge.new_bic_cost -= ComputeBICcost(abs(static_cast<float>(py0)-mrg_py0+1.0));
                ref_edge.new_bic_cost -= ComputeBICcost(abs(static_cast<float>(mrg_py1)-py1+1.0));
                ref_edge.new_bic_cost += ComputeBICcost(abs(static_cast<float>(mrg_py1)-mrg_py0+1.0));

                for(UINT32 kk = 2; kk < 4; kk ++){
                    ref_edge.new_bias[kk] -= ref_bd_sup0.border.border_seg[pt0_key].bias[kk];
                    ref_edge.new_bias[kk] -= ref_bd_sup1.border.border_seg[pt1_key].bias[kk];
                    ref_edge.new_bias[kk] += ref_edge.border.border_seg[mrg_top_key].bias[kk];
                }
            }
        }
    };
    auto ComputeMergeInfo = [&](DistEdge &ref_edge){
        DistSuperPixel &supix0 = m_supixs[ref_edge.sup1];
        DistSuperPixel &supix1 = m_supixs[ref_edge.sup2];
        
        // bbox.
        ref_edge.border.MergeBoundingBox(supix0.border.bbox, supix1.border.bbox);
        
        // construct new region information if do merging.
        ConstructNewRegionInfo(ref_edge);

        // compute mergecost.
        float bias_cost  = 0;
        for(UINT32 k=0; k < e_dist_num_ch; k ++){
            bias_cost += ref_edge.new_bias[k] - (supix0.sum_bias[k] + supix1.sum_bias[k])/2; 
        }
        float sem_diff   = ComputeSemanticDifference(ref_edge.sup1, ref_edge.sup2);
        float sem_cost   = abs(sem_diff - m_pParam->merge_edge_semcost_thr)*(supix0.pixs.size() + supix1.pixs.size());
        float fit_cost   = ref_edge.new_fit_cost - (supix0.sum_fit_cost + supix1.sum_fit_cost);
        float bic_cost   = ref_edge.new_bic_cost - (supix0.sum_bic_cost + supix1.sum_bic_cost);
        float geo_cost   = (float)ref_edge.new_perimeter/(supix0.pixs.size()+supix1.pixs.size()) - 
                            ((float)supix0.perimeter/supix0.pixs.size() + (float)supix1.perimeter/supix1.pixs.size());
        float conn_scale = float(ref_edge.bd_pixs.size())/min(supix0.perimeter, supix1.perimeter); // don't know how to model it in merging cost.

        ref_edge.mergecost = fit_cost + bic_cost + bias_cost*m_pParam->merge_edge_biascost_alpha + 
                                sem_cost*m_pParam->merge_edge_semcost_alpha + geo_cost * m_pParam->merge_edge_geo_alpha;
        if(sem_diff >= m_pParam->merge_edge_semdiff_thr){
            ref_edge.mergecost = m_pParam->merge_edge_inf_cost;
        }
    };
    
    // Main process.
    DistEdge &ref_edge = m_edges[edge];
    if(ref_edge.sup1 == 0 || ref_edge.sup2 ==0)
        return;

    // compute the information if merge the connected two super pixels.
    ref_edge.Reset();
    ComputeMergeInfo(ref_edge);
  
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
    auto FindOppositeX = [&](UINT32 py, UINT32 px, int step){
        UINT32 cx = px;
        while((step<0 && cx > 0) || (step>0 && cx < m_wd-1)){
            cx = cx + step;
            if(m_pInLabelI->GetData(py, cx) != sup){
                cx = cx - step;
                break;
            }   
        }
        return cx;
    };
    auto FindOppositeY = [&](UINT32 py, UINT32 px, int step){
        UINT32 cy = py;
        while((step<0 && cy > 0) || (step>0 && cy < m_ht-1)){
            cy = cy + step;
            if(m_pInLabelI->GetData(cy, px) != sup){
                cy = cy - step;
                break;
            }
        }
        return cy;
    };

    // Main process.
    if(sup == 0)
        return;
    DistSuperPixel &ref_supix = m_supixs[sup];
    
    // compute semantic score.
    ref_supix.sem_score.resize(m_num_sem, 0);
    m_pSemMat->MeanZ(ref_supix.pixs, ref_supix.sem_score);
    
    // compute bounding box.
    UINT32 py, px;
    Border &ref_border = ref_supix.border;
    for(auto it=ref_supix.pixs.begin(); it != ref_supix.pixs.end(); it++){
        py = (*it) / m_wd;   px = (*it) % m_wd;
        ref_border.bbox[0] = min(py, ref_border.bbox[0]);
        ref_border.bbox[1] = min(px, ref_border.bbox[1]);
        ref_border.bbox[2] = max(py, ref_border.bbox[2]);
        ref_border.bbox[3] = max(px, ref_border.bbox[3]);
    }

    // upudate border related information.
    for(auto it : ref_supix.adjacents){
        DistEdge &ref_edge = m_edges[it.second];
        
        // compute perimeter.
        ref_supix.perimeter += ref_edge.bd_pixs.size();

        // find x_h / y_v in on another side.
        for(UINT32 k =0; k < ref_edge.bd_pixs.size(); k++){
            BndryPix &bd_pix = m_borders[ref_edge.bd_pixs[k]];
            if(sup == bd_pix.sup1){
                py = (bd_pix.pix1) / m_wd;   
                px = (bd_pix.pix1) % m_wd;
            }
            else{
                py = (bd_pix.pix2) / m_wd;   
                px = (bd_pix.pix2) % m_wd;
            }

           
            if(bd_pix.pix2 - bd_pix.pix1 == 1){
                UINT32 is_h_line = 1;
                Mkey_3D pt_key(py, px, is_h_line);
                
                // find opposite x_h. and get the segment between them.
                SegFitRst segFit_h;
                if(px > 0 && m_pInLabelI->GetData(py, px-1) == sup){
                    ref_border.border_seg[pt_key].x_h = FindOppositeX(py, px, -1);
                    segFit_h = m_pSegStock->GetAllSegFitResultOnAny2Points(py, ref_border.border_seg[pt_key].x_h, py, px); 
                }
                else if(px < m_wd-1 && m_pInLabelI->GetData(py, px+1) == sup){
                    ref_border.border_seg[pt_key].x_h = FindOppositeX(py, px, 1);
                    segFit_h = m_pSegStock->GetAllSegFitResultOnAny2Points(py, px, py, ref_border.border_seg[pt_key].x_h); 
                }
                else{
                    ref_border.border_seg[pt_key].x_h = px;
                    segFit_h = m_pSegStock->GetAllSegFitResultOnAny2Points(py, px, py, ref_border.border_seg[pt_key].x_h); 
                }
            
                // get segment info on each line.
                ref_border.border_seg[pt_key].bias[0]    = abs(segFit_h.b[0]);
                ref_border.border_seg[pt_key].bias[1]    = abs(segFit_h.b[1]);
                ref_border.border_seg[pt_key].weight[0]  = segFit_h.w[0];
                ref_border.border_seg[pt_key].weight[1]  = segFit_h.w[1];
                ref_border.border_seg[pt_key].fit_err[0] = segFit_h.fit_err;

                // compute statistic fit_err, bic_cost, bias info.
                ref_supix.sum_bias[0] += ref_border.border_seg[pt_key].bias[0];
                ref_supix.sum_bias[1] += ref_border.border_seg[pt_key].bias[1];

                ref_supix.sum_fit_cost += ComputeFitCost(ref_border.border_seg[pt_key], py, px, true);
                ref_supix.sum_bic_cost += ComputeBICcost(abs(static_cast<float>(px) - ref_border.border_seg[pt_key].x_h+1.0));
            }
            else{ 
                UINT32 is_h_line = 0;
                Mkey_3D pt_key(py, px, is_h_line);
                
                // find opposite x_h. y_v, and get the segment between them.
                SegFitRst segFit_v;
                if(py > 0 && m_pInLabelI->GetData(py-1, px) == sup){
                    ref_border.border_seg[pt_key].y_v = FindOppositeY(py, px, -1);
                    segFit_v = m_pSegStock->GetAllSegFitResultOnAny2Points(ref_border.border_seg[pt_key].y_v, px,  py, px); 
                }
                else if(py < m_ht-1 && m_pInLabelI->GetData(py+1, px) == sup){
                    ref_border.border_seg[pt_key].y_v = FindOppositeY(py, px, 1);
                    segFit_v = m_pSegStock->GetAllSegFitResultOnAny2Points(py, px, ref_border.border_seg[pt_key].y_v, px); 
                }
                else{
                    ref_border.border_seg[pt_key].y_v = py;
                    segFit_v = m_pSegStock->GetAllSegFitResultOnAny2Points(py, px, ref_border.border_seg[pt_key].y_v, px); 
                }

                // get segment info on each line.
                ref_border.border_seg[pt_key].bias[2]    = abs(segFit_v.b[0]);
                ref_border.border_seg[pt_key].bias[3]    = abs(segFit_v.b[1]);
                ref_border.border_seg[pt_key].weight[2]  = segFit_v.w[0];
                ref_border.border_seg[pt_key].weight[3]  = segFit_v.w[1];
                ref_border.border_seg[pt_key].fit_err[1] = segFit_v.fit_err;

                // compute statistic fit_err, bic_cost, bias info.
                for(UINT32 kk = 0; kk < e_dist_num_ch; kk ++){
                    ref_supix.sum_bias[kk] += ref_border.border_seg[pt_key].bias[kk];
                }
                ref_supix.sum_fit_cost += ComputeFitCost(ref_border.border_seg[pt_key], py, px, false);
                ref_supix.sum_bic_cost += ComputeBICcost(abs(static_cast<float>(py) - ref_border.border_seg[pt_key].y_v+1.0));
            }
        }
    }
    // cout<< "*** Super Pixel:: "<<sup<<": "<<setw(5)<<ref_supix.pixs.size();
    for(UINT32 kk = 0; kk < e_dist_num_ch; kk ++){
        ref_supix.sum_bias[kk] /= 2;
        // cout<<",    "<<setw(10)<<ref_supix.sum_bias[kk];
    }
    ref_supix.sum_fit_cost /= 2;
    ref_supix.sum_bic_cost /= 2;
    // cout<<",    "<<setw(10)<<ref_supix.sum_fit_cost<<",    "<<setw(10)<<ref_supix.sum_bic_cost<<endl;
    
   

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
}

void SuperPixelMerger::ComputeHistogramW(DistSuperPixel &ref_supix){
    UINT32 bin_w = 0;
    for(auto it: ref_supix.border.border_seg){
        for(UINT32 k=0; k < e_dist_num_ch; k++){
            bin_w = vote2histogram_w(it.second.weight[k]);
            ref_supix.hist_w[k*HIST_W_NUM_BIN+bin_w] += 1;
        }
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
float SuperPixelMerger::ComputeFitCost(BdSeg &ref_bdSeg, float py, float px, bool is_row){
    if(is_row)
        return (ref_bdSeg.fit_err[0] * abs(ref_bdSeg.x_h - px));
    else
        return (ref_bdSeg.fit_err[1] * abs(ref_bdSeg.y_v - py));
}

float SuperPixelMerger::ComputeBICcost(float numPix){
    return (log(numPix + m_pParam->merge_supix_bic_addi_len));
}

float SuperPixelMerger::ComputeSemanticDifference(UINT32 sup0, UINT32 sup1){
    return _ChiDifference(m_supixs[sup0].sem_score, m_supixs[sup1].sem_score);
}


#endif
