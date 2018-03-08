#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentGrow.hxx"
#include "MergerSuperPixel.hxx"


/*
 * Interface to be called by Cython.
 *   input:: imgInfo: size = 4, [imgHt, imgWd, distCh, semCh].
 *           distVec: distance Matrix, size = distCh*imgHt*imgWd. arrange as: first wd, then ht, then distCh.
 *            semVec: semantic Matrix, size = semCh *imgHt*imgWd. arrange as: first wd, then ht, then semCH.
 */
void ProposalGenerate(UINT32* imgInfo, float* distVec, float* semVec, UINT32 * labelArr){
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

    CDataTempl<UINT8> bgSem(imgHt, imgWd);
    semI.Mask(bgSem, 0);
    
    
    // prepare output variable
    UINT32 cnt = 0;
    CDataTempl<UINT32> out_labelI(imgHt, imgWd);

    // global parameter.
    GlbParam glbParam;

    // ----------------------
    // estimate segment from distance map.
    
    std::cout<<"step 1: fitting segment "<<std::endl; 
    Segment_Stock segStock(imgHt, imgWd);
    Segment_Fit segFit(&glbParam, &bgSem, &semI, &semM, &distM); 
    segFit.FittingFeasibleSolution(e_fit_hor, &segStock);
    segFit.FittingFeasibleSolution(e_fit_ver, &segStock);
   
    // case what we are debuging.
#ifdef DEBUG_SEGMENT_STOCK
    out_labelI = segStock.GetSegmentLabelImage();
    for(UINT32 y=0; y < imgHt; y++){ for(UINT32 x=0; x < imgWd; x++){
           labelArr[cnt] = out_labelI.GetDataByIdx(cnt);
           cnt += 1;
        }
    }
    return;
#endif
    
    // ----------------------
    // generate super pixels based on segment.
    std::cout<<"step 2: growing based on segment "<<std::endl; 
    Segment_Grow segGrow(true, &bgSem, &semM, &segStock, &glbParam);
    segGrow.ImagePartition();
    
#ifdef DEBUG_SEGMENT_GROW
    out_labelI = segGrow.GetFinalResult();
    for(UINT32 y=0; y < imgHt; y++){
        for(UINT32 x=0; x < imgWd; x++){
           labelArr[cnt] = out_labelI.GetDataByIdx(cnt);
           cnt += 1;
        }
    }
    return;
#endif

    // ----------------------
    //merge based on generated super pixels.
    std::cout<<"step 3: Merge super pixels. "<<std::endl; 
    SuperPixelMerger supixMerger(&semM, &segStock, &glbParam);
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
    for(UINT32 y=0; y < imgHt; y++){
        for(UINT32 x=0; x < imgWd; x++){
           labelArr[cnt] = out_labelI.GetDataByIdx(cnt);
           cnt += 1;
        }
    }
}








#endif
