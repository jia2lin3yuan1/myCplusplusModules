
#include "SegmentFitting.hxx"
#include "read_write_img.cxx"

void PrepareData(std::string fpath, CDataTempl<double> &distM, CDataTempl<UINT32> &bgSem){
    ReadFromCSV(distM, fpath+"dist0.csv", 0);
    ReadFromCSV(distM, fpath+"dist1.csv", 1);
    ReadFromCSV(distM, fpath+"dist2.csv", 2);
    ReadFromCSV(distM, fpath+"dist3.csv", 3);

    ReadFromCSV(bgSem, fpath+"sem.csv");
}


void testSegmentFitting(std::string fpath, UINT32 imgHt, UINT32 imgWd){
    CDataTempl<double> distM(imgHt, imgWd, 4);
    CDataTempl<UINT32> bgSem(imgHt, imgWd);
    PrepareData(fpath, distM, bgSem);

    bool isRow = true;
    Segment_Fit segFit_H(imgWd);
    segFit_H.AssignY(distM, bgSem, 251, isRow);
    segFit_H.find_keypoints();

    vector<UINT32> iniIdxs = segFit_H.GetIniIdxs();
    for(int k =0; k<iniIdxs.size(); k++){
        std::cout<<iniIdxs[k]<<", ";
    }
    std::cout<<std::endl<<"   ^^^  this is the initial indexes."<<std::endl;


    CDataTempl<double> fit_err(iniIdxs.size(), iniIdxs.size());
    segFit_H.FittingFeasibleSolution(fit_err);
    segFit_H.DP_segments(fit_err);
    
    vector<UINT32> dpIdxs = segFit_H.GetdpIdxs();
    for(int k=0; k<dpIdxs.size(); k++){
        std::cout<<dpIdxs[k]<<", ";
    }
    std::cout<<std::endl<<"   ^^^  this is the dp results."<<std::endl;
}


int main(){
    UINT32 imgHt = 500;
    UINT32 imgWd = 375;
    std::string fpath = "./data_segment/"; 
   
    testSegmentFitting(fpath, imgHt, imgWd);
    
    
    
    return 0;
}
    

