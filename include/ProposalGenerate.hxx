#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentGrow.hxx"
#include "MergerSuperPixel.hxx"
#include "TriMapGenerate.hxx"

template<typename OUT_TYPE>
void ProposalGenerate(CDataTempl<double> &distM, CDataTempl<double> &semM, CDataTempl<UINT32> &instI, CDataTempl<OUT_TYPE> &maskI){
    
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
    
    SuperPixelMerger supixMerger(&semM, &distM, &segStock, &glbParam, &instI);
    supixMerger.AssignInputLabel(&segLabelI);
    supixMerger.CreateGraphFromLabelI();
    supixMerger.ComputeGraphWeights();
    supixMerger.Merger();
    
    vector<double> merge_flag;
    supixMerger.GetMergeInfo(merge_flag);
    cout<<"Merge Falg: size: "<< merge_flag.size()<<endl;
    for(int k =0; k < merge_flag.size(); k += 3){
        cout<<merge_flag[k]<<", "<<merge_flag[k+1]<<", "<<merge_flag[k+2]<<endl;
    }

#ifdef DEBUG_SEGMENT_MERGE
    if (false){
        CDataTempl<double> debugI(imgHt, imgWd);
        supixMerger.GetDebugImage(debugI);
    }
    else
        maskI = supixMerger.AssignOutputLabel();
    return;
#endif
    

#ifdef DEBUG_FINAL_TRIMAP
    // -----------------------
    //generate tri-probability map.
    Trimap_Generate trimapGen(&supixMerger, &segStock, &semM, &distM, &glbParam);
    trimapGen.GreedyGenerateTriMap();
    trimapGen.GetOutputData(maskI);
#else
    maskI = supixMerger.GetSuperPixelIdImage();
#endif

}


#endif
