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

enum Dist_Ch {e_dist_lft=0, e_dist_rht, e_dist_bot, e_dist_top};
enum Fit_Mode {e_fit_hor=0, e_fit_ver};
class Segment_Fit{
protected:
    // inpute
    const CDataTempl<UINT8>  *m_pSem_bg;  // [2D: ht*wd]
    const CDataTempl<UINT32> *m_pSemI;    // [2D: ht*wd]
    const CDataTempl<float>  *m_pSemMat;  // [3D: ht*wd*num_classes]
    const CDataTempl<float>  *m_pDistMat; // [3D: ht*wd*num_dist]
    UINT8 m_fitseg_method;
    
    // generated data, used in whole class.
    UINT32 m_ht,  m_wd;
    UINT32 m_num_sem, m_num_dist;
    CDataTempl<float>     m_lutX;

    // parameter
    const GlbParam     *m_pParam;
    
    
public:
    Segment_Fit(const GlbParam *pParam, const CDataTempl<UINT8> *pSemBg, const CDataTempl<UINT32> *pSemI, const CDataTempl<float> *pSemMat, const CDataTempl<float> *pDistMat, UINT8 fit_method=1){
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
        
        UINT32 lutX_len = max(m_ht, m_wd);
        InitLUTx(lutX_len);
    }
   
    
    // compute fitting error for each combination of start and end point. 
    void FittingFeasibleSolution(Fit_Mode mode, Segment_Stock *pSegStock);
    

protected:
    // prepare data for computing
    void InitLUTx(UINT32 len);

    // find key points given point list.
    void FindKeyPoints(const UINT32 numPt, const vector<UINT32> &ptY, const vector<UINT32> &ptX, UINT32 *dist_ch, vector<UINT32> &out_key_idxs);
    // attribute of segment starting at key_idxs[stK]:
    vector<vector<float> > FitErrorStartFrom(UINT32 dist_ch, float meanY, UINT32 len, UINT32 stK,
                                    const vector<UINT32> &ptY, const vector<UINT32> &ptX, const vector<UINT32> &key_idxs);
    void SemanticScoreStartFrom(auto &semScore, UINT32 len, UINT32 stK, 
                                    const vector<UINT32> &ptY, const vector<UINT32> &ptX, const vector<UINT32> &key_idxs);
    
    // Dynamic Programming to find best partitions of the line.
    void DP_segments(const CDataTempl<float> &seg_info, auto &semScore, const vector<UINT32> &ptY, const vector<UINT32> &ptX,
                      const vector<UINT32> &key_idxs, vector<UINT32> &dp_idxs);
};

void Segment_Fit::InitLUTx(UINT32 len){
    m_lutX.Init(len, 6); // 0-acc_x^2, 1-acc_x, 2-1/(acc_x^2*(x+1) - acc_x*acc_x)
                           // 3-1/(x+1), 4-1/(x^2), 5-x
    m_lutX.SetData(1.0, 0, 3);
    float cumsum_powK = 0, cumsum_K = 0;
    for(UINT32 k = 1; k < len; k ++){
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

void Segment_Fit::SemanticScoreStartFrom(auto &semScore, UINT32 len, UINT32 stK, 
                    const vector<UINT32> &ptY, const vector<UINT32> &ptX, const vector<UINT32> &key_idxs){
    UINT32 idx_len = key_idxs.size();
    vector<float>  acc_sem_score(m_num_sem, 0);
    for(UINT32 k =stK+1; k < idx_len; k ++){
        Mkey_2D key(stK, k);
        semScore[key].resize(m_num_sem, 0);
        UINT32 pt_len = key_idxs[k] - key_idxs[stK]; // length of segments [point_stK, point_k)
        for(UINT32 j=0; j<m_num_sem; j++){
            for(UINT32 i=key_idxs[k-1]; i < key_idxs[k]; i++){
                acc_sem_score[j] += m_pSemMat->GetData(ptY[i], ptX[i], j);
            }
            semScore[key][j] = acc_sem_score[j] / float(pt_len);
        }
    }
}

/* 
 * fitting of segment:
 *     Fitting all segments starting at stK and ending at latter key points
 *     w = (n\sum_{yx}-\sum_y\sum_x) / (n\sum_x^2-\sum_x\sum_x)
 *     b = sum_(y-wx)/n
 *
 *     b = (\sum_y\sum_x^2-\sum_y\sum_x) / (n\sum_x^2-\sum_x\sum_x)
 *     w = sum_(yx-bx)/sum_x^2.
 *
 *     err = sum((wx+b-y)^2) = sum(w^2x^2+2wbx+b^2 - 2*(wxy+by) + y^2)
 *  Input:
 *     dist_ch: channel of distance.
 *     meanY:   meanDist over all points from calling function.
 *     len:     number of points in [point_stK,  point_end]
 *     stK:    
 *     ptY/ptX: all points.
 *     key_idxs: index of key points.
 */
vector<vector<float> > Segment_Fit::FitErrorStartFrom(UINT32 dist_ch, float meanY, UINT32 len, UINT32 stK,
                                            const vector<UINT32> &ptY, const vector<UINT32> &ptX, const vector<UINT32> &key_idxs){
    // compute acc_y, acc_xy, acc_y_2
    vector<float> acc_y(len, 0);
    vector<float> acc_xy(len, 0);
    vector<float> acc_y_2(len, 0);

    float distV = m_pDistMat->GetData(ptY[key_idxs[stK]], ptX[key_idxs[stK]], dist_ch);
    acc_y[0]    = distV;
    acc_xy[0]   = 0;
    acc_y_2[0]  = pow(distV, 2);
    for(UINT32 k =1; k < len; k++){
        distV      = m_pDistMat->GetData(ptY[key_idxs[stK]+k], ptX[key_idxs[stK]+k], dist_ch);
        acc_y[k]   = acc_y[k-1]   + distV;
        acc_xy[k]  = acc_xy[k-1]  + distV*m_lutX.GetData(k, 5);
        acc_y_2[k] = acc_y_2[k-1] + pow(distV, 2);
    }

    // compute w and b for segments starting at stK and ending at latter key points
    UINT32 idx_len = key_idxs.size();
    vector<vector<float> > fit_err(idx_len, vector<float>(3, 0.0));
    for(UINT32 k =stK+1; k < idx_len; k ++){
        UINT32 tk = key_idxs[k] - key_idxs[stK]; // length of segments [point_stK, point_k]
        float w, b;
        if (m_fitseg_method == 1){
            b = (acc_y[tk]*m_lutX.GetData(tk, 0) - acc_xy[tk]*m_lutX.GetData(tk, 1)) * m_lutX.GetData(tk, 2);
            w = (acc_xy[tk] - b*m_lutX.GetData(tk, 1)) * m_lutX.GetData(tk, 4);
        }
        else{
            w = (acc_xy[tk]*(m_lutX.GetData(tk, 5)+1) - acc_y[tk]*m_lutX.GetData(tk, 1))*m_lutX.GetData(tk,2);
            b = (acc_y[tk] - w*m_lutX.GetData(tk, 1)) * m_lutX.GetData(tk, 3);
        }

        float term_0  = w*w*m_lutX.GetData(tk, 0) + 2*w*b*m_lutX.GetData(tk, 1) + b*b*(m_lutX.GetData(tk, 5)+1);
        float term_1  = 2*(w*acc_xy[tk] + b*acc_y[tk]);
        float err_sum = term_0 - term_1 + acc_y_2[tk];
        err_sum       = err_sum>=0? err_sum : -err_sum;
        fit_err[k][0] = err_sum/max(min(acc_y[tk], meanY * (m_lutX.GetData(tk, 5)+1)), float(1e-9));
        fit_err[k][1] = w;
        fit_err[k][2] = b;
    }

    return fit_err;
}

/*
 * compute fitting error for combination of any two key points.[st, end] 
 */
void Segment_Fit::FittingFeasibleSolution(Fit_Mode mode, Segment_Stock *pSegStock){
    auto ComputeMeanDistance = [&](UINT32 *dist_ch, UINT32 numPt, vector<UINT32> &ptY, vector<UINT32> &ptX, float *meanD){
        meanD[0] = 0;   meanD[1] = 0;
        for(UINT32 k=0; k < numPt; k++){
            meanD[0] += m_pDistMat->GetData(ptY[k], ptX[k], dist_ch[0]);
            meanD[1] += m_pDistMat->GetData(ptY[k], ptX[k], dist_ch[1]);
        }

        meanD[0] /= numPt;
        meanD[1] /= numPt;
    };
   
    // Main process.
    // prepare data.
    UINT32 num_line, num_pt, dist_ch[2];
    vector<UINT32> ptYs(m_ht+1, 0);
    vector<UINT32> ptXs(m_wd+1, 0);
    if(mode == e_fit_hor){
        num_line   = m_ht;   
        num_pt     = m_wd;
        dist_ch[0] = e_dist_lft;
        dist_ch[1] = e_dist_rht;

        for(UINT32 k=0; k<=m_wd; k++)
            ptXs[k] = k; 
    }
    else{ // e_fit_ver
        num_line   = m_wd;   
        num_pt     = m_ht;
        dist_ch[0] = e_dist_bot;
        dist_ch[1] = e_dist_top;
        
        for(UINT32 k=0; k<=m_ht; k++)
            ptYs[k] = k; 
    }

    // start process for each line
    for(UINT32 k=0; k < num_line; k++){
        if(mode==e_fit_hor)
            ptYs.assign(m_wd+1, k);
        else
            ptXs.assign(m_ht+1, k);

        // prepare.
        float meanD[2];
        ComputeMeanDistance(dist_ch, num_pt, ptYs, ptXs, meanD);
        vector<UINT32> key_idxs;
        FindKeyPoints(num_pt, ptYs, ptXs, dist_ch, key_idxs);
       
        // compute information for each small segment.
        UINT32 num_key = key_idxs.size();
        CDataTempl<float> segInfo(num_key, num_key, 5);
        map<Mkey_2D, vector<float>, MKey2DCmp>      semScore;
        map<Mkey_2D, map<string, float>, MKey2DCmp> fitResult;
        for(UINT32 j=0; j < num_key-1; j++){
            UINT32 len_2end = num_pt - key_idxs[j];
            vector<vector<float> > fit_err_0 = FitErrorStartFrom(dist_ch[0], meanD[0], len_2end, j, ptYs, ptXs, key_idxs); 
            vector<vector<float> > fit_err_1 = FitErrorStartFrom(dist_ch[1], meanD[1], len_2end, j, ptYs, ptXs, key_idxs); 
            SemanticScoreStartFrom(semScore, len_2end, j, ptYs, ptXs, key_idxs);
            for(UINT32 i=j+1; i<key_idxs.size(); i++){
                segInfo.SetData(fit_err_0[i][0]+fit_err_1[i][0], j, i, 0);
                segInfo.SetData(fit_err_0[i][1], j, i, 1);
                segInfo.SetData(fit_err_0[i][2], j, i, 2);
                segInfo.SetData(fit_err_1[i][1], j, i, 3);
                segInfo.SetData(fit_err_1[i][2], j, i, 4);
            }
        }

        // DP to cut line into optimal partitions.
        vector<UINT32> dp_idxs;
        DP_segments(segInfo, semScore, ptYs, ptXs, key_idxs, dp_idxs);

        // Save segment information to segment stock.
        pSegStock->AssignAllSegments(segInfo, key_idxs, ptYs, ptXs, dist_ch);
        pSegStock->AssignDpSegments(segInfo, semScore, dp_idxs, ptYs, ptXs);
    }
}

/* 
 * find key points given Y0 and Y1.
 *    input:  dist_ch, which 2 channels are looking into in distance mat.
 */
void Segment_Fit::FindKeyPoints(const UINT32 numPt, const vector<UINT32> &ptY, const vector<UINT32> &ptX, UINT32 *dist_ch, 
                                vector<UINT32> &out_key_idxs){
    auto IsKeyPoint = [&](UINT32 k){
        UINT32 cy = ptY[k],   cx = ptX[k];
        // background point could not be a key point
        if(m_pSem_bg->GetData(cy, cx) == 1)
            return false;

        UINT32 ny = ptY[k+1], nx = ptX[k+1];
        UINT32 py = ptY[k-1], px = ptX[k-1];
        // check semantic background.
        if(m_pSem_bg->GetData(cy, cx)==0 && (m_pSem_bg->GetData(py, px)==1 || m_pSem_bg->GetData(ny, nx)==1))
            return true;

        // check semantic label.
        if(m_pSemI->GetData(cy,cx) != m_pSemI->GetData(py, px))
            return true;

        // check distance channels.
        for(UINT32 i=0; i < 2; i++){
            UINT32 ch = dist_ch[i];
            if((m_pDistMat->GetData(cy, cx, ch)-m_pDistMat->GetData(py, px, ch)) * (m_pDistMat->GetData(cy, cx, ch)-m_pDistMat->GetData(ny, nx, ch))>0)
                return true;
        }

        return false;
    };
    
    
    // Main process. 
    // segments are [st, end), "0" and "numPt-1" are default to be key points.
    assert(out_key_idxs.size()==0);
    out_key_idxs.insert(out_key_idxs.end(), 0);
   
    // check points [1, numPt-2]
    for(UINT32 k=1; k < numPt-1; k ++){
        if(IsKeyPoint(k)){
            out_key_idxs.insert(out_key_idxs.end(), k);
        }
    }
    
    out_key_idxs.insert(out_key_idxs.end(), numPt-1);
}

/*
 * Dynamic Programming to find best partitions of the line.
 */
void Segment_Fit::DP_segments(const CDataTempl<float> &seg_info, auto &semScore, const vector<UINT32> &ptY, const vector<UINT32> &ptX,
                      const vector<UINT32> &key_idxs, vector<UINT32> &dp_idxs){
    auto NewSegmentFromSemantic = [&](UINT32 k, UINT32 st_k, UINT32 len){
        // semantic background
        UINT32 py = (ptY[key_idxs[k]] + ptY[key_idxs[k-1]])/2;
        UINT32 px = (ptX[key_idxs[k]] + ptX[key_idxs[k-1]])/2;
        if(m_pSem_bg->GetData(py, px) == 1)
            return 1;
    
        // semantic score diff
        if(k < len-1){
            Mkey_2D obsKey(k, k+1);
            Mkey_2D expKey(st_k, k);
            float diff = _ChiDifference(semScore[obsKey], semScore[expKey]);
            if(diff > m_pParam->segFit_dp_semdiff_thr)
                return 2;
        }

        return 0;
    };

    // forward-backward recordings:
    UINT32 len  = key_idxs.size();
    vector<float>  acc_costV(len, 0);
    vector<float>  seg_costV(len, 0);
    vector<int>    acc_numV(len, 0);
    vector<int>    bk_routeV(len, 0);

    // forward.
    UINT32 st_k = 0;
    for(UINT32 k = 1; k < len; k++){
        // check if the semantic information implies starting of a new segment.
        UINT32 semantic_type = NewSegmentFromSemantic(k, st_k, len);
        if(semantic_type == 1){
           st_k          = k;
           acc_costV[k]  = 0;
           seg_costV[k]  = 0;
           acc_numV[k]   = 0;
           bk_routeV[k]  = k-1;
           continue;
        }
        
        // traverse all availabe selection, find the optimal solution
        UINT32 min_idx = 0;
        float min_cost = 2*m_pParam->segFit_dp_inf_err, min_fitCost=0, min_numSeg=0;
        for(UINT32 j = st_k; j < k; j ++){
            // compute cost for having segment [iniIdxs[st_k],  iniIdxs[j])
            // float BIC_cost  = (acc_numV[j]+1) * Log_LUT(m_iniIdxs[k]-m_iniIdxs[st_k]+1);
            float BIC_cost  = (acc_numV[j]+1) * log(key_idxs[k]-key_idxs[st_k]+1);
            float fit_cost  = seg_costV[j] + seg_info.GetData(j, k, 0)*(key_idxs[k]-key_idxs[j]);
            float fit_resist = m_pParam->segFit_dp_inf_err * (seg_info.GetData(j, k,0) > m_pParam->segFit_dp_err_thr);
            float cand_cost  = fit_cost + m_pParam->segFit_dp_bic_alpha*BIC_cost + fit_resist;

            if(cand_cost < min_cost){
                min_cost    = cand_cost;
                min_fitCost = fit_cost;
                min_numSeg  = acc_numV[j] + 1;
                min_idx     = j;
            }
        }
        bk_routeV[k] = min_idx;
        
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
    UINT32 ci  = len-1;
    while(ci > 0){
        dp_idxs.insert(dp_idxs.begin(), key_idxs[ci]);
        dp_idxs.insert(dp_idxs.begin(), ci);
        ci = bk_routeV[ci];
    }
    dp_idxs.insert(dp_idxs.begin(), key_idxs[ci]);
    dp_idxs.insert(dp_idxs.begin(), ci);
}

#endif
