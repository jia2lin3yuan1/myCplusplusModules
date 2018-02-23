#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentFitting.hxx"
#include "SegmentGrow.hxx"

void SegmentFittingOneLine(Segment_Fit &segFit, Segment_Grow &segGrow,  CDataTempl<double> &distM, CDataTempl<UINT32> &bgSem, UINT32 line, bool isRow){
    // First, find the best partition of a line into several small segments. 
    segFit.AssignY(distM, bgSem, line, isRow);
    segFit.find_keypoints();
    vector<UINT32> iniIdxs = segFit.GetIniIdxs();
    
    CDataTempl<double> fit_err(iniIdxs.size(), iniIdxs.size());
    segFit.FittingFeasibleSolution(fit_err);
    
    segFit.DP_segments(fit_err);
    vector<UINT32> dpIdxs = segFit.GetdpIdxs();
    
    // Record the status into global segment_grow. 
    segGrow.AssignAllSegment(fit_err, iniIdxs, line, isRow);
    if(isRow)
        segGrow.AssignDPSegment_H(bgSem, fit_err, dpIdxs, line);
    else
        segGrow.AssignDPSegment_V(bgSem, fit_err, dpIdxs, line);
}




void ProposalGenerate(CDataTempl<double> &distM, CDataTempl<UINT32> &bgSem, CDataTempl<UINT32> &maskI){
    
    UINT32 imgHt = distM.GetYDim();
    UINT32 imgWd = distM.GetXDim();

    Segment_Grow segGrow(imgHt, imgWd);

    Segment_Fit segFit_H(imgWd);
    for(UINT32 k=0; k < imgHt; ++k){
        SegmentFittingOneLine(segFit_H, segGrow, distM, bgSem, k, true);
    }

    Segment_Fit segFit_V(imgHt);
    for(UINT32 k=0; k < imgWd; ++k){
        SegmentFittingOneLine(segFit_V, segGrow, distM, bgSem, k, false);
    }

    segGrow.ImagePartition(bgSem);
    maskI = segGrow.GetFinalResult();
}








#endif
