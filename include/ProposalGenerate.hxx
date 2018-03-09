#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentGrow.hxx"
#include "MergerSuperPixel.hxx"


template<typename OUT_TYPE>
void ProposalGenerate(CDataTempl<float> &distM, CDataTempl<float> &semM, CDataTempl<OUT_TYPE> &maskI){
    
    UINT32 imgHt = distM.GetYDim();
    UINT32 imgWd = distM.GetXDim();
    
    CDataTempl<UINT32> semI(imgHt, imgWd);
    semM.argmax(semI, 2);

    CDataTempl<UINT8> bgSem(imgHt, imgWd);
    semI.Mask(bgSem, 0);

    GlbParam glbParam;

    // ----------------------
    // estimate segment from distance map.
    Segment_Stock segStock(imgHt, imgWd);
    Segment_Fit segFit(&glbParam, &bgSem, &semI, &semM, &distM); 
    segFit.FittingFeasibleSolution(e_fit_hor, &segStock);
    segFit.FittingFeasibleSolution(e_fit_ver, &segStock);


    // case what we are debuging.
#ifdef DEBUG_SEGMENT_STOCK
    maskI = segStock.GetSegmentLabelImage();
    return;
#endif
    // ----------------------
    // generate super pixels based on segment.
    Segment_Grow segGrow(true, &bgSem, &semM, &segStock, &glbParam);
    segGrow.ImagePartition();
    
#ifdef DEBUG_SEGMENT_GROW
    maskI = segGrow.GetFinalResult();
    return;

#endif
    
    // ----------------------
    //merge based on generated super pixels.
    CDataTempl<UINT32> segLabelI;
    segLabelI = segGrow.GetFinalResult();
    
    SuperPixelMerger supixMerger(&semM, &segStock, &glbParam);
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


    // -----------------------
    // generate tri-probability map.

}


#endif
