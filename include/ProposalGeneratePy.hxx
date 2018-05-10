#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentGrow.hxx"
#include "MergerSuperPixel.hxx"
#include "TriMapGenerate.hxx"


/*
 * Interface to be called by Cython.
 *   input:: imgInfo: size = 4, [imgHt, imgWd, distCh, semCh].
 *           distVec: distance Matrix, size = distCh*imgHt*imgWd. arrange as: first wd, then ht, then distCh.
 *            semVec: semantic Matrix, size = semCh *imgHt*imgWd. arrange as: first wd, then ht, then semCH.
 */
std::vector<double> ProposalGenerate(UINT32* imgInfo, double* distVec, double* semVec){
    // load data to CDataTemplate from vector.
    UINT32 imgHt  = imgInfo[0];
    UINT32 imgWd  = imgInfo[1];
    UINT32 distCh = imgInfo[2];
    UINT32 semCh  = imgInfo[3];

    std::cout<<"Image info is: ht/wd = "<< imgHt << " / " << imgWd << ", dist/sem ch = "<<distCh<<" / "<<semCh<<std::endl;

    CDataTempl<double> distM(imgHt, imgWd, distCh);
    distM.AssignFromVector(distVec);
    
    CDataTempl<double> semM(imgHt, imgWd, semCh);
    semM.AssignFromVector(semVec);
   
    CDataTempl<UINT32> semI(imgHt, imgWd);
    semM.argmax(semI, 2);

    CDataTempl<UINT8> bgSem(imgHt, imgWd);
    semI.Mask(bgSem, 0);
    
    
    // prepare output variable
    UINT32 cnt = 0;

#ifdef DEBUG_FINAL_TRIMAP
    CDataTempl<double> out_labelI;
#else
    CDataTempl<UINT32> out_labelI;
#endif
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
#else
    // ----------------------
    // generate super pixels based on segment.
    std::cout<<"step 2: growing based on segment "<<std::endl; 
    Segment_Grow segGrow(true, &bgSem, &semM, &segStock, &glbParam);
    segGrow.ImagePartition();
    
#ifdef DEBUG_SEGMENT_GROW
    out_labelI = segGrow.GetFinalResult();
#else

    // ----------------------
    //merge based on generated super pixels.
    std::cout<<"step 3: Merge super pixels. "<<std::endl; 
    SuperPixelMerger supixMerger(&semM, &distM, &segStock, &glbParam, &semI);
    CDataTempl<UINT32> segLabelI;
    segLabelI = segGrow.GetFinalResult();
    supixMerger.AssignInputLabel(&segLabelI);
    supixMerger.CreateGraphFromLabelI();
    supixMerger.ComputeGraphWeights();

    /*
    CDataTempl<double> debugI(imgHt, imgWd);
    supixMerger.GetDebugImage(debugI);
    maskI = debugI;
    */
    supixMerger.Merger();

    
#ifdef DEBUG_FINAL_TRIMAP
    // ----------------------------
    // generate trimap for different instance proposals.
    std::cout<<"step 4: Generate TriMap. "<<std::endl; 
    Trimap_Generate trimapGen(&supixMerger, &segStock, &semM, &distM, &glbParam);
    trimapGen.GreedyGenerateTriMap();
    std::cout<<"Growing done."<<std::endl;
    trimapGen.GetOutputData(out_labelI);
#else
    out_labelI = supixMerger.AssignOutputLabel();
#endif // Trimap
#endif // Grow
#endif // Stock

    // ------------------------------------
    // CDataTemplate to vector.
    UINT32 outCh = out_labelI.GetZDim();
    std::vector<double> out_vec(imgHt*imgWd*outCh, 0);
    for(UINT32 z=0; z < outCh; z++){
        for(UINT32 y=0; y < imgHt; y++){
            for(UINT32 x=0; x < imgWd; x++){
               out_vec[cnt] = double(out_labelI.GetDataByIdx(cnt));
               cnt += 1;
            }
        }
    }

    return out_vec;
}








#endif
