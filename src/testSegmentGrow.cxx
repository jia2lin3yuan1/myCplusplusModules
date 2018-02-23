
#include "ProposalGenerate.hxx"
#include "read_write_img.cxx"

/*
 * Before test this case, pass the test of segment fitting first.
 *
 */


void PrepareData(std::string fpath, CDataTempl<double> &distM, CDataTempl<UINT32> &bgSem){
    ReadFromCSV(distM, fpath+"dist0.csv", 0);
    ReadFromCSV(distM, fpath+"dist1.csv", 1);
    ReadFromCSV(distM, fpath+"dist2.csv", 2);
    ReadFromCSV(distM, fpath+"dist3.csv", 3);

    ReadFromCSV(bgSem, fpath+"sem.csv");
}


void testSegmentGrow(std::string fpath, UINT32 imgHt, UINT32 imgWd){
    CDataTempl<double> distM(imgHt, imgWd, 4);
    CDataTempl<UINT32> bgSem(imgHt, imgWd);
    PrepareData(fpath, distM, bgSem);

    CDataTempl<UINT32> maskI;
    ProposalGenerate(distM, bgSem, maskI);
    WriteToCSV(maskI, "test.csv");
    
}


int main(){
    UINT32 imgHt = 500;
    UINT32 imgWd = 375;
    std::string fpath = "./data_segment/"; 
   
    testSegmentGrow(fpath, imgHt, imgWd);
    
    
    
    return 0;
}
    

