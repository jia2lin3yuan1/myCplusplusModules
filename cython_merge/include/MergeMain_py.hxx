#ifndef PROPOSAL_GENERATE_HXX
#define PROPOSAL_GENERATE_HXX

#include "MergerSuperPixel.hxx"

class OutData{
public:
    vector<double> labels;
    OutData(){
        labels.resize(0, 0);
    }
};

/*
 * Interface to be called by Cython.
 *   input:: imgInfo: size = 4, [imgHt, imgWd, distCh, semCh].
 *           distVec: distance Matrix, size = distCh*imgHt*imgWd. arrange as: first wd, then ht, then distCh.
 *            semVec: semantic Matrix, size = semCh *imgHt*imgWd. arrange as: first wd, then ht, then semCH.
 */
void ProposalGenerate(double* imgInfo, double* predVec, int *oversegVec, OutData *outdata){
    // load data to CDataTemplate from vector.
    int imgHt           = int(imgInfo[0]);
    int imgWd           = int(imgInfo[1]);
    int num_obj         = int(imgInfo[2]);
    double obj_cen_maxV = imgInfo[3];

    cout<<"Image info is: ht/wd = "<< imgHt << " / " << imgWd <<endl;

    CDataTempl<double> predM(imgHt, imgWd);
    predM.AssignFromVector(predVec);

    CDataTempl<int> oversegI(imgHt, imgWd);
    oversegI.AssignFromVector(oversegVec);
    
    vector<double> merge_flag;
    CDataTempl<int> out_labelI;
    
    // prepare output variable

    // global parameter.
    GlbParam glbParam(obj_cen_maxV);
    //GlbParam glbParam(10000);
    //num_obj = 1000; 
    
    
    // ----------------------
    //merge based on generated super pixels.
    double pred_clip =20;
    SuperPixelMerger supixMerger(pred_clip, &predM, &glbParam);
    supixMerger.AssignInputLabel(&oversegI);
    supixMerger.CreateGraphFromLabelI();
    supixMerger.ComputeGraphProperties();
    supixMerger.Merger(num_obj+1);

    out_labelI = supixMerger.AssignOutputLabel();

    // ------------------------------------
    // CDataTemplate to vector.
    int cnt = 0;
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
