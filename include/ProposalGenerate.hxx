#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentGrow.hxx"
#include "MergerSuperPixel.hxx"

void SegmentFittingOneLine(Segment_Fit &segFit, Segment_Stock &segStock,  CDataTempl<float> &distM, CDataTempl<UINT32> &bgSem, UINT32 line, bool isRow){
    // First, find the best partition of a line into several small segments. 
    segFit.AssignY(distM, bgSem, line, isRow);
    segFit.FindKeyPoints();
    vector<UINT32> iniIdxs = segFit.GetIniIdxs();
    
    CDataTempl<float> fit_err(iniIdxs.size(), iniIdxs.size());
    segFit.FittingFeasibleSolution(fit_err);
    
    segFit.DP_segments(fit_err);
    vector<UINT32> dpIdxs = segFit.GetdpIdxs();
    
    // Record the status into global segment_grow. 
    segStock.AssignAllSegments(fit_err, iniIdxs, line, isRow);
    segStock.AssignDpSegments(fit_err, dpIdxs, line, isRow);
}



template<typename OUT_TYPE>
void ProposalGenerate(CDataTempl<float> &distM, CDataTempl<UINT32> &bgSem, CDataTempl<OUT_TYPE> &maskI){
    
    UINT32 imgHt = distM.GetYDim();
    UINT32 imgWd = distM.GetXDim();

    GlbParam glbParam;

    // ----------------------
    // estimate segment from distance map.
    Segment_Stock segStock(imgHt, imgWd);
    Segment_Fit segFit_H(imgWd, &glbParam);
    for(UINT32 k=0; k < imgHt; ++k){
        SegmentFittingOneLine(segFit_H, segStock, distM, bgSem, k, true);
    }

    Segment_Fit segFit_V(imgHt, &glbParam);
    for(UINT32 k=0; k < imgWd; ++k){
        SegmentFittingOneLine(segFit_V, segStock, distM, bgSem, k, false);
    }
   
    // case what we are debuging.
#ifdef DEBUG_SEGMENT_STOCK
    maskI = segStock.GetSegmentLabelImage();
    return;
#endif
    // ----------------------
    // generate super pixels based on segment.
    Segment_Grow segGrow(imgHt, imgWd, true, bgSem, &segStock, &glbParam);
    segGrow.ImagePartition(bgSem);
    
#ifdef DEBUG_SEGMENT_GROW
    maskI = segGrow.GetFinalResult();
    return;
#endif

    // ----------------------
    //merge based on generated super pixels.
    CDataTempl<UINT32> segLabelI;
    segLabelI = segGrow.GetFinalResult();

    SuperPixelMerger supixMerger(imgHt, imgWd, &segStock, &glbParam);
    supixMerger.AssignInputLabel(&segLabelI);
    supixMerger.CreateGraphFromLabelI();
    supixMerger.ComputeGraphWeights();

#ifdef DEBUG_SEGMENT_MERGE_DEBUG
    CDataTempl<float> debugI(imgHt, imgWd);
    supixMerger.GetDebugImage(debugI);
    maskI = debugI;
#endif
    
    supixMerger.Merger();
    maskI = supixMerger.AssignOutputLabel();
}








#endif
