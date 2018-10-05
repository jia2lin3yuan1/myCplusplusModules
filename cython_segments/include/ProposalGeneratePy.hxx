#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentGrow.hxx"
#include "MergerSuperPixel.hxx"
#include "TriMapGenerate.hxx"

class OutData{
public:
    vector<double> labels;
    vector<double> merge_flag;
    OutData(){
        labels.resize(0, 0);
        merge_flag.resize(0, 0);
    }
};

/*
 * Interface to be called by Cython.
 *   input:: imgInfo: size = 4, [imgHt, imgWd, distCh, semCh].
 *           distVec: distance Matrix, size = distCh*imgHt*imgWd. arrange as: first wd, then ht, then distCh.
 *            semVec: semantic Matrix, size = semCh *imgHt*imgWd. arrange as: first wd, then ht, then semCH.
 */
void ProposalGenerate(int* imgInfo, double* distVec, double* semVec, int *instVec, OutData *outdata){
    // load data to CDataTemplate from vector.
    int imgHt  = imgInfo[0];
    int imgWd  = imgInfo[1];
    int distCh = imgInfo[2];
    int semCh  = imgInfo[3];

    cout<<"Image info is: ht/wd = "<< imgHt << " / " << imgWd << ", dist/sem ch = "<<distCh<<" / "<<semCh<<endl;

    CDataTempl<double> distM(imgHt, imgWd, distCh);
    distM.AssignFromVector(distVec);
    
    CDataTempl<double> semM(imgHt, imgWd, semCh);
    semM.AssignFromVector(semVec);
   
    CDataTempl<int> semI(imgHt, imgWd);
    semM.argmax(semI, 2);

    CDataTempl<int> bgSem(imgHt, imgWd);
    semI.Mask(bgSem, 0);

    CDataTempl<int> instI(imgHt, imgWd);
    instI.AssignFromVector(instVec);
    
    vector<double> merge_flag;
    
    // prepare output variable
    int cnt = 0;

#ifdef DEBUG_FINAL_TRIMAP
    CDataTempl<double> out_labelI;
#else
    CDataTempl<int> out_labelI;
#endif
    // global parameter.
    GlbParam glbParam;

    // ----------------------
    // estimate segment from distance map.
    
    cout<<"step 1: fitting segment "<<endl; 
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
    cout<<"step 2: growing based on segment "<<endl; 
    Segment_Grow segGrow(true, &bgSem, &semM, &segStock, &glbParam);
    segGrow.ImagePartition();
    
#ifdef DEBUG_SEGMENT_GROW
    out_labelI = segGrow.GetFinalResult();
#else

    // ----------------------
    //merge based on generated super pixels.
    cout<<"step 3: Merge super pixels. "<<endl; 
    SuperPixelMerger supixMerger(&semM, &distM, &segStock, &glbParam, &instI);
    CDataTempl<int> segLabelI;
    segLabelI = segGrow.GetFinalResult();
    supixMerger.AssignInputLabel(&segLabelI);
    supixMerger.CreateGraphFromLabelI();
    supixMerger.ComputeGraphWeights();
    supixMerger.Merger();
    
    supixMerger.GetMergeInfo(merge_flag);
    outdata->merge_flag = merge_flag;

    
#ifdef DEBUG_FINAL_TRIMAP
    // ----------------------------
    // generate trimap for different instance proposals.
    cout<<"step 4: Generate TriMap. "<<endl; 
    Trimap_Generate trimapGen(&supixMerger, &segStock, &semM, &distM, &glbParam);
    trimapGen.GreedyGenerateTriMap();
    cout<<"Growing done."<<endl;
    trimapGen.GetOutputData(out_labelI);
#else
    out_labelI = supixMerger.AssignOutputLabel();
#endif // Trimap
#endif // Grow
#endif // Stock

    // ------------------------------------
    // CDataTemplate to vector.
    int outCh = out_labelI.GetZDim();
    vector<double> out_vec(imgHt*imgWd*outCh, 0);
    for(int z=0; z < outCh; z++){
        for(int y=0; y < imgHt; y++){
            for(int x=0; x < imgWd; x++){
               out_vec[cnt] = double(out_labelI.GetDataByIdx(cnt));
               cnt += 1;
            }
        }
    }

    outdata->labels = out_vec;
    cout<<"Finished.  "<<endl; 
}








#endif
