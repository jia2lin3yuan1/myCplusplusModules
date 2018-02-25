#ifndef SEGMENT_FITTING_HXX
#define SEGMENT_FITTING_HXX

#include "DataTemplate.hxx"
#include "LogLUT.hxx"
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

enum Dist_Ch {dist_lft=0, dist_rht, dist_bot, dist_top};

class Segment_Fit{
protected:
    UINT32                m_len;
    CDataTempl<UINT32>    m_Ysem;
    CDataTempl<double>    m_Y0;
    CDataTempl<double>    m_Y1;
    
    double                m_meanY0;
    double                m_meanY1;
    CDataTempl<double>    m_lutX;

    // output
    vector<UINT32>        m_iniIdxs;
    vector<UINT32>        m_dpIdxs;

    // parameter
    const GlbParam     *m_pParam;
public:
    Segment_Fit(UINT32 len, const GlbParam *pParam){
        m_len = len;
        InitLUTx();
        m_Y0.Init(m_len);
        m_Y1.Init(m_len);
        m_Ysem.Init(m_len);

        m_pParam = pParam;
    }
    
    void InitLUTx();

    void AssignY(CDataTempl<double> &matY, CDataTempl<UINT32> &matYsem, UINT32 k, bool isRow=true);
    vector<UINT32>& GetIniIdxs(){return m_iniIdxs;}
    vector<UINT32>& GetdpIdxs(){return m_dpIdxs;}

    // fitting of segment starting at m_iniIdxs[stK]:
    vector<double> FittingStartPoint(CDataTempl<double> &Y, double meanY, UINT32 stK, UINT8 mtd=1);

    // compute fitting error for each combination of start and end point. 
    void FittingFeasibleSolution(CDataTempl<double> &fit_err);

    // find key points given Y0 and Y1.
    void FindKeyPoints();
    
    // Dynamic Programming to find best partitions of the line.
    void DP_segments(CDataTempl<double> &fit_err);

};

void Segment_Fit::InitLUTx(){
    m_lutX.Init(m_len, 6); // 0-acc_x^2, 1-acc_x, 2-1/(acc_x^2*(x+1) - acc_x*acc_x)
                           // 3-1/(x+1), 4-1/(x^2), 5-x
    m_lutX.SetData(1.0, 0, 3);
    double cumsum_powK = 0, cumsum_K = 0;
    for(UINT32 k = 1; k < m_len; k ++){
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

void Segment_Fit::AssignY(CDataTempl<double> &matY, CDataTempl<UINT32> &matYsem, UINT32 k, bool isRow){
    if (isRow){
        matY.GetRow(m_Y0, k, dist_lft);
        matY.GetRow(m_Y1, k, dist_rht);
        matYsem.GetRow(m_Ysem, k);
    }
    else{
        matY.GetColumn(m_Y0, k, dist_bot);
        matY.GetColumn(m_Y1, k, dist_top);
        matYsem.GetColumn(m_Ysem, k);
    }

    m_meanY0 = max(m_Y0.Mean(), 1e-6);
    m_meanY1 = max(m_Y1.Mean(), 1e-6);
    
    m_iniIdxs.clear();
    m_dpIdxs.clear();
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
 */
vector<double> Segment_Fit::FittingStartPoint(CDataTempl<double> &Y, double meanY, UINT32 stK, UINT8 mtd){
    // compute acc_y, acc_xy, acc_y_2
    UINT32 len = m_len - m_iniIdxs[stK];
    vector<double> acc_y(len, 0);
    vector<double> acc_xy(len, 0);
    vector<double> acc_y_2(len, 0);
    acc_y[0]   = Y.GetData(m_iniIdxs[stK]);
    acc_y_2[0] = pow(Y.GetData(m_iniIdxs[stK]), 2);
    for(UINT32 k =1; k < len; k++){
        acc_y[k]  = acc_y[k-1] + Y.GetData(m_iniIdxs[stK]+k);
        acc_xy[k] = acc_xy[k-1] + Y.GetData(m_iniIdxs[stK]+k)*m_lutX.GetData(k, 5);
        acc_y_2[k] = acc_y_2[k-1] + pow(Y.GetData(m_iniIdxs[stK]+k), 2);
    }

    // compute w and b for segments starting at stK and ending at latter key points
    UINT32 idx_len = m_iniIdxs.size();
    vector<double> fit_err(idx_len, 0.);
    for(UINT32 k =stK+1; k < idx_len; k ++){
        UINT32 tk = m_iniIdxs[k]-m_iniIdxs[stK]; // length of segments [iniIdxs[stK], iniIdxs[k] ]
        double w, b;
        if (mtd == 1){
            b = (acc_y[tk]*m_lutX.GetData(tk, 0) - acc_xy[tk]*m_lutX.GetData(tk, 1)) * m_lutX.GetData(tk, 2);
            w = (acc_xy[tk] - b*m_lutX.GetData(tk, 1)) * m_lutX.GetData(tk, 4);
        }
        else if (mtd==2){
            w = (acc_xy[tk]*(m_lutX.GetData(tk, 5)+1) - acc_y[tk]*m_lutX.GetData(tk, 1))*m_lutX.GetData(tk,2);
            b = (acc_y[tk] - w*m_lutX.GetData(tk, 1)) * m_lutX.GetData(tk, 3);
        }

        double term_0  = w*w*m_lutX.GetData(tk, 0) + 2*w*b*m_lutX.GetData(tk, 1) + b*b*(m_lutX.GetData(tk, 5)+1);
        double term_1  = 2*(w*acc_xy[tk] + b*acc_y[tk]);
        double err_sum = term_0 - term_1 + acc_y_2[tk];
        err_sum        = err_sum>=0? err_sum : -err_sum;
        fit_err[k]     = err_sum/min(max(acc_y[tk], 1e-9), meanY*(m_lutX.GetData(tk, 5)+1));
    }

    return fit_err;
}

/*
 * compute fitting error for combination of any two key points.[st, end] 
 */
void Segment_Fit::FittingFeasibleSolution(CDataTempl<double> &fit_err){
   UINT32 len = m_iniIdxs.size();
   for(UINT32 k = 0; k < len-1; k ++){
       vector<double> fit_err_0 = FittingStartPoint(m_Y0, m_meanY0, k);
       vector<double> fit_err_1 = FittingStartPoint(m_Y1, m_meanY1, k);
        
       for(UINT32 j=k+1; j < len; j++){
           fit_err.SetData(fit_err_0[j]+fit_err_1[j], k, j);
       }
   }
}

/* 
 * find key points given Y0 and Y1.
 */
void Segment_Fit::FindKeyPoints(){
    // segments are [st, end), "0" and "m_len" are default to be key points.
    m_iniIdxs.insert(m_iniIdxs.end(), 0);
    
    for(UINT32 k=1; k < m_len-1; k ++){
        // check if the point is a key point.
        bool is_key_pt = false;
        if((m_Y0.GetData(k)-m_Y0.GetData(k-1))*(m_Y0.GetData(k)-m_Y0.GetData(k+1))>0)
            is_key_pt = true;
        
        if((m_Y1.GetData(k)-m_Y1.GetData(k-1))*(m_Y1.GetData(k)-m_Y1.GetData(k+1))>0)
            is_key_pt = true;
        
        if(m_Ysem.GetData(k)==0 && (m_Ysem.GetData(k-1)>0 || m_Ysem.GetData(k+1)>0))
            is_key_pt = true;
       
        // if it is key point, add to container.
        if(is_key_pt == true)
            m_iniIdxs.insert(m_iniIdxs.end(), k);
    }

    m_iniIdxs.insert(m_iniIdxs.end(), m_len-1);
    
}

/*
 * Dynamic Programming to find best partitions of the line.
 */
void Segment_Fit::DP_segments(CDataTempl<double> &fit_err){
    // forward-backward recordings:
    UINT32 len  = m_iniIdxs.size();
    vector<double> acc_costV(len, 0);
    vector<double> seg_costV(len, 0);
    vector<int>    acc_numV(len, 0);
    vector<int>    bk_routeV(len, 0);

    // forward.
    UINT32 st_k = 0;
    for(UINT32 k = 1; k < len; k++){
        // if the previous side is background, it must be cutted to a segment.
        if(m_Ysem.GetData(m_iniIdxs[k]-1)){
           st_k = k;
           acc_costV[k] = 0;
           seg_costV[k] = 0;
           acc_numV[k]  = 0;
           bk_routeV[k] = k-1;
        }
        else{
            UINT32 min_idx = 0;
            double min_cost = 2*m_pParam->segFit_inf_err, min_fitCost=0, min_numSeg=0;
            for(UINT32 j = st_k; j < k; j ++){
                // compute cost for having segment [iniIdxs[st_k],  iniIdxs[j]]
                double BIC_cost  = (acc_numV[j]+1) * Log_LUT(m_iniIdxs[k]-m_iniIdxs[st_k]+1);
                double fit_cost  = seg_costV[j] + fit_err.GetData(j, k)*(m_iniIdxs[k]-m_iniIdxs[j]);
                double fit_resist = m_pParam->segFit_inf_err*(fit_err.GetData(j, k)>m_pParam->segFit_err_thr);
                double cand_cost  = fit_cost + m_pParam->segFit_bic_alpha*BIC_cost + fit_resist;

                if(cand_cost < min_cost){
                    min_cost    = cand_cost;
                    min_fitCost = fit_cost;
                    min_numSeg  = acc_numV[j] + 1;
                    min_idx     = j;
                }
            }

            // assigning to the optimal solution
            acc_costV[k] = min_cost;
            seg_costV[k] = min_fitCost;
            acc_numV[k]  = min_numSeg;
            bk_routeV[k] = min_idx;
        }
    }

    // backward.
    UINT32 ci  = len-1;
    while(ci > 0){
        m_dpIdxs.insert(m_dpIdxs.begin(), m_iniIdxs[ci]);
        m_dpIdxs.insert(m_dpIdxs.begin(), ci);
        ci = bk_routeV[ci];
    }
    m_dpIdxs.insert(m_dpIdxs.begin(), m_iniIdxs[ci]);
    m_dpIdxs.insert(m_dpIdxs.begin(), ci);
}

#endif
