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


/*
 * Interface to be called by Cython.
 *   input:: imgInfo: size = 4, [imgHt, imgWd, distCh, semCh].
 *           distVec: distance Matrix, size = distCh*imgHt*imgWd. arrange as: first wd, then ht, then distCh.
 *            semVec: semantic Matrix, size = semCh *imgHt*imgWd. arrange as: first wd, then ht, then semCH.
 */
void ProposalGenerate(std::vector<UINT32> & imgInfo, const std::vector<float> & distVec, const std::vector<float> &semVec, UINT32 * labelArr){
    // load data to CDataTemplate from vector.
    UINT32 imgHt  = imgInfo[0];
    UINT32 imgWd  = imgInfo[1];
    UINT32 distCh = imgInfo[2];
    UINT32 semCh  = imgInfo[3];

    std::cout<<"Image info is: ht/wd = "<< imgHt << " / " << imgWd << ", dist/sem ch = "<<distCh<<" / "<<semCh<<std::endl;
    
    CDataTempl<float> distM(imgHt, imgWd, distCh);
    distM.AssignFromVector(distVec);
    
    CDataTempl<float> semM(imgHt, imgWd, semCh);
    semM.AssignFromVector(semVec);
   
    CDataTempl<UINT32> semI(imgHt, imgWd);
    semM.argmax(semI, 2);

    CDataTempl<UINT32> bgSem(imgHt, imgWd);
    semI.Equal(bgSem, 0);
    
    
    // prepare output variable
    CDataTempl<UINT32> out_labelI(imgHt, imgWd);

    // global parameter.
    GlbParam glbParam;

    // ----------------------
    // estimate segment from distance map.
    
    std::cout<<"step 1: fitting segment "<<std::endl; 
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
    out_labelI = segStock.GetSegmentLabelImage();
    return;
#endif
    
    // ----------------------
    // generate super pixels based on segment.
    std::cout<<"step 2: growing based on segment "<<std::endl; 
    Segment_Grow segGrow(imgHt, imgWd, bgSem, &segStock, &glbParam);
    segGrow.ImagePartition(bgSem);
    
#ifdef DEBUG_SEGMENT_GROW
    out_labelI = segGrow.GetFinalResult();
    return;
#endif

    // ----------------------
    //merge based on generated super pixels.
    std::cout<<"step 3: Merge super pixels. "<<std::endl; 
    SuperPixelMerger supixMerger(imgHt, imgWd, &segStock, &glbParam);
    CDataTempl<UINT32> segLabelI;
    segLabelI = segGrow.GetFinalResult();
    supixMerger.AssignInputLabel(&segLabelI);
    supixMerger.CreateGraphFromLabelI();
    supixMerger.ComputeGraphWeights();

    /*
    CDataTempl<float> debugI(imgHt, imgWd);
    supixMerger.GetDebugImage(debugI);
    maskI = debugI;
    */
    supixMerger.Merger();
    out_labelI = supixMerger.AssignOutputLabel();

    // CDataTemplate to vector.
    UINT32 cnt = 0;
    for(UINT32 y=0; y < imgHt; y++){
        for(UINT32 x=0; x < imgWd; x++){
           labelArr[cnt] = out_labelI.GetDataByIdx(cnt);
           cnt += 1;
        }
    }
}








#endif
