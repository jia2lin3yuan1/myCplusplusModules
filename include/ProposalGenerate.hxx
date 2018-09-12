#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "SegmentGrow.hxx"
#include "MergerSuperPixel.hxx"
#include "TriMapGenerate.hxx"

void FilterBilateral(double dist0[], double dist1[], double out_coef[]){
    double coef[2];
    coef[0] = 0.7;
    coef[1] = (1-coef[0])/2;
    if(dist0[1] == 0 || dist1[1]==0){
        out_coef[0] = coef[1];
        out_coef[1] = coef[0];
        out_coef[2] = coef[1];
    }

    out_coef[1] = coef[0];
    if((dist0[0] > 0 && dist0[0] <= dist0[1]) || (dist1[0]>0 && dist1[0] >= dist1[1])){
        out_coef[0]= coef[1];
    }
    else{
        out_coef[1] += coef[1];
    }
    if((dist0[2] > 0 && dist0[2] >= dist0[1]) || (dist1[2]>0 && dist1[2] <= dist1[1])){
        out_coef[2]= coef[1];
    }
    else{
        out_coef[1] += coef[1];
    }
}

void DistancePreprocessing(CDataTempl<double> &distM){
    int imgHt = distM.GetYDim();
    int imgWd = distM.GetXDim();
    double dist0[3], dist1[3];
    double fil_coef[3];

    // Horizontal distance.
    for(int y=0; y<imgHt; y++){
        int y_n1 = y==0? y : y-1;
        int y_p1 = y==imgHt-1? y: y+1;
        for(int x=0; x<imgWd; x++){
            dist0[0] = distM.GetData(y_n1, x, 2);
            dist0[1] = distM.GetData(y,    x, 2);
            dist0[2] = distM.GetData(y_p1, x, 2);
            dist1[0] = distM.GetData(y_n1, x, 3);
            dist1[1] = distM.GetData(y,    x, 3);
            dist1[2] = distM.GetData(y_p1, x, 3);
            FilterBilateral(dist0, dist1, fil_coef);
            for(int ch=0; ch< 2; ch++){
                double fil_var = distM.GetData(y_n1, x, ch)*fil_coef[0];
                fil_var += distM.GetData(y,x,ch)*fil_coef[1];
                fil_var += distM.GetData(y_p1, x, ch)*fil_coef[2];
                distM.SetData(fil_var, y, x, ch);
            }

        }
    }

    // Vertical distance.
    for(int x=0; x<imgWd; x++){
        int x_n1 = x==0? x : x-1;
        int x_p1 = x==imgWd-1? x: x+1;
        for(int y=0; y<imgHt; y++){
            dist0[0] = distM.GetData(y, x_n1, 2);
            dist0[1] = distM.GetData(y, x, 2);
            dist0[2] = distM.GetData(y, x_p1, 2);
            dist1[0] = distM.GetData(y, x_n1, 3);
            dist1[1] = distM.GetData(y, x, 3);
            dist1[2] = distM.GetData(y, x_p1, 3);
            FilterBilateral(dist0, dist1, fil_coef);
            for(int ch=2; ch< 4; ch++){
                double fil_var = distM.GetData(y, x_n1, ch)*fil_coef[0];
                fil_var += distM.GetData(y,x,ch)*fil_coef[1];
                fil_var += distM.GetData(y, x_p1, ch)*fil_coef[2];
                distM.SetData(fil_var, y, x, ch);
            }

        }
    }
}


template<typename OUT_TYPE>
void ProposalGenerate(CDataTempl<double> &distM, CDataTempl<double> &semM, CDataTempl<int> &instI, CDataTempl<OUT_TYPE> &maskI){
    
    int imgHt = distM.GetYDim();
    int imgWd = distM.GetXDim();
    
    CDataTempl<int> semI(imgHt, imgWd);
    semM.argmax(semI, 2);

    CDataTempl<int> bgSem(imgHt, imgWd);
    semI.Mask(bgSem, 0);

    GlbParam glbParam;

    // ----------------------
    // estimate segment from distance map.
    DistancePreprocessing(distM);

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
    CDataTempl<int> segLabelI;
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
