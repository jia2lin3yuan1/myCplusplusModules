#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "MergerSuperPixel.hxx"

template<typename OUT_TYPE>
void ProposalGenerate(int num_obj, double obj_cen_maxV, CDataTempl<double> &predM, CDataTempl<int> &oversegI, CDataTempl<OUT_TYPE> &maskI){
    
    GlbParam glbParam(obj_cen_maxV);

    // ----------------------
    CDataTempl<int> segLabelI;
    double pred_clip = 20; 
    SuperPixelMerger supixMerger(pred_clip, &predM, &glbParam);
    supixMerger.AssignInputLabel(&oversegI);
    supixMerger.CreateGraphFromLabelI();
    supixMerger.ComputeGraphProperties();
    cout<<"--- start merging "<<endl;
    supixMerger.Merger(num_obj + 1);
    cout<<"--- finished "<<endl;
    
    if (false){
        int imgHt = predM.GetYDim();
        int imgWd = predM.GetXDim();
        CDataTempl<double> debugI(imgHt, imgWd);
        supixMerger.GetDebugImage(debugI);
    }
    else
        maskI = supixMerger.AssignOutputLabel();
    return;
}


#endif
