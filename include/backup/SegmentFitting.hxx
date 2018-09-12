#ifndef SEGMENT_FITTING_HXX
#define SEGMENT_FITTING_HXX

#include "utils/DataTemplate.hxx"
#include "utils/LogLUT.hxx"
#include "SegmentStock.hxx"

using namespace std;

/*
 * Class: Segment_Fit:
 *  It is used to find the optimal partition of a line given estimated distance in 2 opposed direction.
 *  Input: dist_Y0 / dist_Y1: distances
 *         sem_bgY : background flag of the line.
 *
 *  Functions: FindKeyPoints() : find index of points that gradient of distances, or sem_bg flag got changed.
 *             FittingFeasibleSolution(out_fit_err): compute error of fitting linear segment between any two key points.
 *             DP_segments(in_fit_err): 
 */
// Create of class Segment_Fit.
// Target: Given a 1D vector, find the best way to fit it into several sections.
//

enum Dist_Ch {e_dist_lft=0, e_dist_rht, e_dist_bot, e_dist_top, e_dist_num_ch};
enum Fit_Mode {e_fit_hor=0, e_fit_ver, e_fit_num};
class Segment_Fit{
protected:
    // inpute
    const CDataTempl<int>  *m_pSem_bg;  // [2D: ht*wd]
    const CDataTempl<int> *m_pSemI;    // [2D: ht*wd]
    const CDataTempl<double>  *m_pSemMat;  // [3D: ht*wd*num_classes]
    const CDataTempl<double>  *m_pDistMat; // [3D: ht*wd*num_dist]
    int m_fitseg_method;
    
    // generated data, used in whole class.
    int m_ht,  m_wd;
    int m_num_sem, m_num_dist;
    CDataTempl<double>     m_lutX;

    // parameter
    const GlbParam     *m_pParam;
    
    
public:
    Segment_Fit(const GlbParam *pParam, const CDataTempl<int> *pSemBg, const CDataTempl<int> *pSemI, const CDataTempl<double> *pSemMat, const CDataTempl<double> *pDistMat, int fit_method=1){
        m_pParam   = pParam;
        m_pSem_bg  = pSemBg;
        m_pSemI    = pSemI;
        m_pSemMat  = pSemMat;
        m_pDistMat = pDistMat;
        m_fitseg_method = fit_method;

        m_ht       = m_pSem_bg->GetYDim();
        m_wd       = m_pSem_bg->GetXDim();
        m_num_sem  = m_pSemMat->GetZDim(); 
        m_num_dist = m_pDistMat->GetZDim(); 
        
        int lutX_len = max(m_ht, m_wd);
        InitLUTx(lutX_len);
    }
   
    
    // compute fitting error for each combination of start and end point. 
    void FittingFeasibleSolution(Fit_Mode mode, Segment_Stock *pSegStock);
    

protected:
    // prepare data for computing
    void InitLUTx(int len);

    // find all possible segments and compute their information.
    void ComputeFittingMap(const auto &ptY, const auto &ptX, int ch, bool invPt, 
                            auto &key_idxs, auto &key_type, auto &seg_semscore, auto &seg_fitinfo);
    // fitting error for a segment with length 'len' and given y related acc info.
    vector<double> FitSegmentError(int len, double acc_y, double acc_xy, double acc_y2, double meanY);
    
    // Dynamic Programming to find best partitions of the line.
    void DP_segments(auto &key_idxs, auto &key_type, auto &seg_semscore, auto &seg_fitinfo, bool invPt, vector<int> &dp_idxs);
};

void Segment_Fit::InitLUTx(int len){
    m_lutX.Init(len, 6); // 0-acc_x^2, 1-acc_x, 2-1/(acc_x^2*(x+1) - acc_x*acc_x)
                           // 3-1/(x+1), 4-1/(acc_x^2), 5-x
    m_lutX.SetData(1.0, 0, 3);
    double cumsum_powK = 0, cumsum_K = 0;
    for(int k = 1; k < len; k ++){
        cumsum_K    += k;
        cumsum_powK += pow(k, 2); 
        m_lutX.SetData(cumsum_powK, k, 0);
        m_lutX.SetData(cumsum_K,  k, 1);
        m_lutX.SetData(1.0/(k+1), k, 3);
        m_lutX.SetData(k, k, 5);

        m_lutX.SetData(1.0/(cumsum_powK*(k+1) - cumsum_K*cumsum_K), k, 2);
        m_lutX.SetData(1.0/cumsum_powK, k, 4);
    }
}

/*
 * generate fitting map for every possible line segments.
 * output: vector[int] key_idxs. [0, key_i  ,numPt-1]
 *         vector[int] key_type. denote the type (distance is 0 or not) of segment starting from each possible key_idx.
 *         map<Mkey_2D, vector<doutle>, MKey2DCmp>:      average semantic score for each segment
 *         map<Mkey_2D, map<string, double>, MKey2DCmp>: fitting result
 */
void Segment_Fit::ComputeFittingMap(const auto &ptY, const auto &ptX, int ch, bool invPt, 
                        auto &key_idxs, auto &key_type, auto &seg_semscore, auto &seg_fitinfo){
    auto ComputeMeanDistance = [&](int numPt){
        double meanD = 0;
        for(int k=0; k < numPt; k++){
            meanD += m_pDistMat->GetData(ptY[k], ptX[k], ch);
        }
        
        return (meanD/numPt);
    };
    auto addSemanticScoreOnePoint = [&](auto &sem_vec, int pt_i){
        for(int j=0; j<m_num_sem; j++){
            sem_vec[j] += m_pSemMat->GetData(ptY[pt_i], ptX[pt_i], j);
        }
    };
    auto minusSemanticScoreOneSeg = [&](auto &sum_sem, Mkey_2D cut_key, Mkey_2D add_key){
        seg_semscore[add_key].resize(m_num_sem, 0);
        for(int j=0; j<m_num_sem; j++){
            seg_semscore[add_key][j] = sum_sem[j] - seg_semscore[cut_key][j];
        }
    };

    // Main process.
    int    numPt    = ptY.size();
    double meanD    = ComputeMeanDistance(numPt);
    int    st       = invPt?  numPt-1 : 0;
    int    end      = invPt?  0 : numPt-1;
    int    step     = invPt? -1 : 1;
    
    double distV_n1 = 0;
    double distV    = m_pDistMat->GetData(ptY[st], ptX[st], ch);
    int keyT = distV==0? 1 : 0;
    key_type.push_back(keyT);
    key_idxs.push_back(st);
    
    // local variables.
    vector<double> sum_sem(m_num_sem, 0);
    map<int, map<string, double>> seg_acc_yinfo;
    int key_st  = 0;
    seg_acc_yinfo[0]["acc_y"]  = 0;
    seg_acc_yinfo[0]["acc_xy"] = 0;
    seg_acc_yinfo[0]["acc_y2"] = 0;
    for(int k=1; k<numPt; k++){
        int pt_k = k*step + st;
        
        // update infomation.
        distV_n1 = distV;
        addSemanticScoreOnePoint(sum_sem, pt_k-step);
        if(k==numPt-1){
            addSemanticScoreOnePoint(sum_sem, pt_k);
        }
        for(auto it=seg_acc_yinfo.begin(); it!=seg_acc_yinfo.end(); it++){
            if(it->first < key_st){
                continue;
            }
            else{
                it->second["acc_y"]   += distV_n1;
                it->second["acc_y2"]  += pow(distV_n1, 2);
                it->second["acc_xy"]  += distV_n1 * abs(pt_k-(it->first*step+st));
            }
            if(k==numPt-1){
                it->second["acc_y"]   += distV;
                it->second["acc_y2"]  += pow(distV, 2);
                it->second["acc_xy"]  += distV * (abs(pt_k-(it->first*step+st))+1);
            }
        }
        distV = m_pDistMat->GetData(ptY[pt_k], ptX[pt_k], ch);
        
        // if current status is in background area.
        if(key_type.back()==1 && (distV > 0 || k==numPt-1)){
            // add of one segment.
            Mkey_2D key(key_idxs.back(), pt_k);
            seg_fitinfo[key]["error"] = 0;
            seg_fitinfo[key]["w"]     = 0;
            seg_fitinfo[key]["b"]     = 0;
            seg_semscore[key]        = sum_sem;
            
            // prepare for next segment.
            key_st = k;
            sum_sem.assign(sum_sem.size(), 0);
            key_idxs.push_back(pt_k);
            key_type.push_back(distV==0? 1 : 0);
            seg_acc_yinfo[k]["acc_y"]  = 0;
            seg_acc_yinfo[k]["acc_xy"] = 0;
            seg_acc_yinfo[k]["acc_y2"] = 0;
        }
        // if current is inside distance area.
        else if(key_type.back()==0 && (distV < distV_n1 || k==numPt-1)){
            // add of possible segments
            for(auto it : seg_acc_yinfo){
                if(it.first < key_st){
                    continue;
                }
                // compute segment.
                double acc_y  = it.second["acc_y"];
                double acc_xy = it.second["acc_xy"];
                double acc_y2 = it.second["acc_y2"];
                int len_seg   = k == numPt-1? pt_k-(it.first*step+st) : pt_k-(it.first*step+st)-1;
                vector<double> fit_info = FitSegmentError(len_seg, acc_y, acc_xy, acc_y2, meanD);
                Mkey_2D key(it.first*step+st, pt_k);
                seg_fitinfo[key]["error"] = fit_info[0];
                seg_fitinfo[key]["w"]     = fit_info[1];
                seg_fitinfo[key]["b"]     = fit_info[2];
                if(it.first == key_st)
                    seg_semscore[key] = sum_sem;
                else{
                    Mkey_2D cut_key(key_st*step+st, it.first*step+st);
                    minusSemanticScoreOneSeg(sum_sem, cut_key, key);
                }
            }
            
            // prepare for next segment.
            key_idxs.push_back(pt_k);
            if(distV == 0){
                key_st = k;
                sum_sem.assign(sum_sem.size(), 0);
                key_type.push_back(1);
            }
            else{
                key_type.push_back(0);
            }
            seg_acc_yinfo[k]["acc_y"]  = 0;
            seg_acc_yinfo[k]["acc_xy"] = 0;
            seg_acc_yinfo[k]["acc_y2"] = 0;
        }
    }

    // compute average value for semantic score.
    for(auto it=seg_semscore.begin(); it!=seg_semscore.end(); it++){
        int seg_len = it->first.id1 - it->first.id0;
        for(int j=0; j<m_num_sem; j++){
            it->second[j] = it->second[j]/static_cast<double>(seg_len);
        }
    }
}


/* 
 * fitting of segment:
 *     Fitting all segments starting at stK and ending at latter key points
 *     w = (n\sum_{yx}-\sum_y\sum_x) / (n\sum_x^2-\sum_x\sum_x)
 *     b = sum_(y-wx)/n
 *
 *     b = (\sum_y\sum_x^2-\sum_{yx}\sum_x) / (n\sum_x^2-\sum_x\sum_x)
 *     w = sum_(yx-bx)/sum_x^2.
 *
 *     err = sum((wx+b-y)^2) = sum(w^2x^2+2wbx+b^2 - 2*(wxy+by) + y^2)
 *  Input:
 *     len  :   number of points in [point_stK,  point_end]
 *     acc_y, acc_xy, acc_y_2
 *     meanY:   meanDist over all points from calling function.
 *  Output:
 *     fit_err,  w,  b
 */
vector<double> Segment_Fit::FitSegmentError(int len, double acc_y, double acc_xy, double acc_y2, double meanY){
    vector<double> fit_rst(3, 0);
    if(acc_y == 0){
        return fit_rst;
    }

    // fit w, b:  y = wx + b.
    double w, b;
    if (m_fitseg_method == 1){
        b = (acc_y*m_lutX.GetData(len, 0) - acc_xy*m_lutX.GetData(len, 1)) * m_lutX.GetData(len, 2);
        w = (acc_xy-b*m_lutX.GetData(len, 1)) * m_lutX.GetData(len, 4);
    }
    else{
        w = (acc_xy*(m_lutX.GetData(len, 5)+1) - acc_y*m_lutX.GetData(len, 1))*m_lutX.GetData(len,2);
        b = (acc_y-w*m_lutX.GetData(len, 1)) * m_lutX.GetData(len, 3);
    }
    
    // compute error
    double term_0  = w*w*m_lutX.GetData(len, 0) + 2*w*b*m_lutX.GetData(len, 1) + b*b*(m_lutX.GetData(len, 5)+1);
    double term_1  = 2*(w*acc_xy + b*acc_y);
    double err_sum = term_0 - term_1 + acc_y2;
    err_sum       = err_sum>=0? err_sum : -err_sum;

    // return
    fit_rst[0] = err_sum/min(acc_y, meanY*(m_lutX.GetData(len, 5)+1));
    fit_rst[1] = w;
    fit_rst[2] = b;
    
    return fit_rst;
}

/*
 * compute fitting error for combination of any two key points.[st, end] 
 */
void Segment_Fit::FittingFeasibleSolution(Fit_Mode mode, Segment_Stock *pSegStock){
    // prepare data.
    int num_line, num_pt, dist_ch[2];
    vector<int> ptYs(m_ht, 0);
    vector<int> ptXs(m_wd, 0);
    if(mode == e_fit_hor){
        num_line   = m_ht;   
        dist_ch[0] = e_dist_lft;
        dist_ch[1] = e_dist_rht;

        for(int k=0; k<m_wd; k++)
            ptXs[k] = k; 
    }
    else{ // e_fit_ver
        num_line   = m_wd;   
        dist_ch[0] = e_dist_bot;
        dist_ch[1] = e_dist_top;
        
        for(int k=0; k<m_ht; k++)
            ptYs[k] = k; 
    }

    // start process for each line
    for(int k=0; k < num_line; k++){
        if(mode==e_fit_hor)
            ptYs.assign(m_wd, k);
        else
            ptXs.assign(m_ht, k);

        // compute segment in each single channel
        vector<int> dp_idxs_0;
        vector<int> key_idxs_0;
        vector<int> key_type_0;
        map<Mkey_2D, vector<double>, MKey2DCmp> seg_semscore_0;
        map<Mkey_2D, map<string, double>, MKey2DCmp> seg_fitinfo_0;
        ComputeFittingMap(ptYs, ptXs, dist_ch[0], false, key_idxs_0, key_type_0, seg_semscore_0, seg_fitinfo_0);
        DP_segments(key_idxs_0, key_type_0, seg_semscore_0, seg_fitinfo_0, false, dp_idxs_0); 
      
        vector<int> dp_idxs_1;
        vector<int> key_idxs_1;
        vector<int> key_type_1;
        map<Mkey_2D, vector<double>, MKey2DCmp> seg_semscore_1;
        map<Mkey_2D, map<string, double>, MKey2DCmp> seg_fitinfo_1;
        ComputeFittingMap(ptYs, ptXs, dist_ch[0], true, key_idxs_1, key_type_1, seg_semscore_1, seg_fitinfo_1);
        DP_segments(key_idxs_1, key_type_1, seg_semscore_1, seg_fitinfo_1, true, dp_idxs_1); 


        // Save segment information to segment stock.
        pSegStock->AssignAllSegments(seg_fitinfo_1, key_idxs_1, ptYs, ptXs, dist_ch);
        pSegStock->AssignDpSegments(seg_fitinfo_1, seg_semscore_1, dp_idxs_1, ptYs, ptXs);
    }
}

/*
 * Dynamic Programming to find best partitions of the line.
 * input:  vector<int> key_idxs. [0, key_i  ,numPt-1]
 *         vector<int> key_type. denote the type (distance is 0 or not) of segment starting from each possible key_idx.
 *         map<Mkey_2D, vector<doutle>, MKey2DCmp>:      average semantic score list for each segment
 *         map<Mkey_2D, map<string, double>, MKey2DCmp>: fitting result
 *
 * output: vector<int> dp_idxs. [0, key_i, numPt-1]
 */
void Segment_Fit::DP_segments(auto &key_idxs, auto &key_type, auto &seg_semscore, auto &seg_fitinfo, bool invPt,
                                vector<int> &dp_idxs){
    auto NewSegmentFromSemantic = [&](int k, int len){
        // previous seg has distance 0.
        if(key_type[k-1] == 1)
            return 1;
    
        // semantic score diff
        if(k < len-1){
            Mkey_2D obsKey(key_idxs[k], key_idxs[k+1]);
            Mkey_2D expKey(key_idxs[k-1], key_idxs[k]);
            double diff = _ChiDifference(seg_semscore[obsKey], seg_semscore[expKey]);
            if(diff > m_pParam->segFit_dp_semdiff_thr)
                return 2;
        }

        return 0;
    };

    // forward-backward recordings:
    int len  = key_idxs.size();
    vector<double>  acc_costV(len, 0);
    vector<double>  seg_costV(len, 0);
    vector<int>    acc_numV(len, 0);
    vector<int>    bk_routeV(len, 0);

    // forward.
    int st_k = 0;
    for(int k = 1; k < len; k++){
        // check if the semantic information implies starting of a new segment.
        int semantic_type = NewSegmentFromSemantic(k, len);
        if(semantic_type == 1){
           st_k          = k;
           acc_costV[k]  = 0;
           seg_costV[k]  = 0;
           acc_numV[k]   = 0;
           bk_routeV[k]  = k-1;
           continue;
        }
        
        // traverse all availabe selection, find the optimal solution
        int min_idx = 0;
        double min_cost = 2*m_pParam->segFit_dp_inf_err, min_fitCost=0, min_numSeg=0;
        double len_stk = key_idxs[k]-key_idxs[st_k] + 1;
        for(int j = st_k; j < k; j ++){
            // compute cost for having segment [iniIdxs[st_k],  iniIdxs[j])
            Mkey_2D key(key_idxs[j], key_idxs[k]);
            double BIC_cost  = (acc_numV[j]+1) * len_stk;
            double fit_cost  = seg_costV[j] + seg_fitinfo[key]["error"]*(key_idxs[k]-key_idxs[j]);
            double fit_resist = m_pParam->segFit_dp_inf_err * (seg_fitinfo[key]["error"] > m_pParam->segFit_dp_err_thr);
            double cand_cost  = fit_cost + m_pParam->segFit_dp_bic_alpha*BIC_cost + fit_resist;

            if(cand_cost < min_cost){
                min_cost    = cand_cost;
                min_fitCost = fit_cost;
                min_numSeg  = acc_numV[j] + 1;
                min_idx     = j;
            }
        }
        bk_routeV[k] = min_idx;
        
        // if segment [k-1, k) and [k, k+1) has different semantic type, reset forward record for segment [k, k+1)
        if(semantic_type == 2){
           st_k         = k;
           acc_costV[k] = 0;
           seg_costV[k] = 0;
           acc_numV[k]  = 0;
        }
        else{
            acc_costV[k] = min_cost;
            seg_costV[k] = min_fitCost;
            acc_numV[k]  = min_numSeg;
        }
    }

    // backward.
    if(invPt){
        int ci  = len-1;
        while(ci > 0){
            dp_idxs.insert(dp_idxs.end(), len-1-ci);
            dp_idxs.insert(dp_idxs.end(), key_idxs[ci]);
            ci = bk_routeV[ci];
        }
        dp_idxs.insert(dp_idxs.end(), len-1-ci);
        dp_idxs.insert(dp_idxs.end(), key_idxs[ci]);
    }
    else{
        int ci  = len-1;
        while(ci > 0){
            dp_idxs.insert(dp_idxs.begin(), key_idxs[ci]);
            dp_idxs.insert(dp_idxs.begin(), ci);
            ci = bk_routeV[ci];
        }
        dp_idxs.insert(dp_idxs.begin(), key_idxs[ci]);
        dp_idxs.insert(dp_idxs.begin(), ci);
    }
}

#endif
