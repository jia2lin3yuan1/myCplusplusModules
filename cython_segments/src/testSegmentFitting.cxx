
#include "SegmentFitting.hxx"
#include "utils/read_write_img.hxx"

void PrepareData(std::string fpath, CDataTempl<float> &distM, CDataTempl<int> &bgSem){
    ReadFromCSV(distM, fpath+"dist0.csv", 0);
    ReadFromCSV(distM, fpath+"dist1.csv", 1);
    ReadFromCSV(distM, fpath+"dist2.csv", 2);
    ReadFromCSV(distM, fpath+"dist3.csv", 3);

    ReadFromCSV(bgSem, fpath+"sem.csv");
}

void testOneDirection(Segment_Fit &segFit, CDataTempl<float> &distM, CDataTempl<int> &bgSem, int line, bool isRow){
    std::cout<< " *** Line "<<line<<" :"<<std::endl; 
    
    segFit.AssignY(distM, bgSem, line, isRow);
    segFit.FindKeyPoints();
    vector<int> iniIdxs = segFit.GetIniIdxs();
    for(int k =0; k<iniIdxs.size(); k++){
        std::cout<<iniIdxs[k]<<", ";
    }
    std::cout<<std::endl<<"   ^^^  this is the initial indexes."<<std::endl;


    CDataTempl<float> fit_err(iniIdxs.size(), iniIdxs.size());
    segFit.FittingFeasibleSolution(fit_err);
    segFit.DP_segments(fit_err);
    vector<int> dpIdxs = segFit.GetdpIdxs();
    for(int k=0; k<dpIdxs.size(); k++){
        std::cout<<dpIdxs[k]<<", ";
    }
    std::cout<<std::endl<<"   ^^^  this is the dp results."<<std::endl;
}

void testSegmentFitting(std::string fpath, int imgHt, int imgWd){
    CDataTempl<float> distM(imgHt, imgWd, 4);
    CDataTempl<int> bgSem(imgHt, imgWd);
    PrepareData(fpath, distM, bgSem);

    GlbParam glbParam;
    Segment_Fit segFit_H(imgWd, &glbParam);
    testOneDirection(segFit_H, distM, bgSem, 251, true);
    testOneDirection(segFit_H, distM, bgSem, 252, true);
   
    std::cout<<std::endl<<"********** Test Vertical: "<<std::endl;
    Segment_Fit segFit_V(imgHt, &glbParam);
    testOneDirection(segFit_V, distM, bgSem, 100, false);
    testOneDirection(segFit_V, distM, bgSem, 101, false);
}


int main(){
    int imgHt = 500;
    int imgWd = 375;
    std::string fpath = "./input/"; 
   
    testSegmentFitting(fpath, imgHt, imgWd);
    
    
    
    return 0;
}
    

