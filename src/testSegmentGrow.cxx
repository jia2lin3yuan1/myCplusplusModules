
#include "ProposalGenerate.hxx"
#include "utils/read_write_img.hxx"

/*
 * Before test this case, pass the test of segment fitting first.
 *
 */


void PrepareData(std::string fpath, CDataTempl<float> &distM, CDataTempl<float> &semM, CDataTempl<UINT32> &instI){
    ReadFromCSV(distM, fpath+"dist0.csv", 0);
    ReadFromCSV(distM, fpath+"dist1.csv", 1);
    ReadFromCSV(distM, fpath+"dist2.csv", 2);
    ReadFromCSV(distM, fpath+"dist3.csv", 3);

    ReadFromCSV(semM, fpath+"sem0.csv", 0);
    ReadFromCSV(semM, fpath+"sem1.csv", 1);
    ReadFromCSV(semM, fpath+"sem2.csv", 2);
    ReadFromCSV(semM, fpath+"sem3.csv", 3);
    ReadFromCSV(semM, fpath+"sem4.csv", 4);
    
    ReadFromCSV(semM, fpath+"sem5.csv", 5);
    ReadFromCSV(semM, fpath+"sem6.csv", 6);
    ReadFromCSV(semM, fpath+"sem7.csv", 7);
    ReadFromCSV(semM, fpath+"sem8.csv", 8);
    ReadFromCSV(semM, fpath+"sem9.csv", 9);
    
    ReadFromCSV(semM, fpath+"sem10.csv", 10);
    ReadFromCSV(semM, fpath+"sem11.csv", 11);
    ReadFromCSV(semM, fpath+"sem12.csv", 12);
    ReadFromCSV(semM, fpath+"sem13.csv", 13);
    ReadFromCSV(semM, fpath+"sem14.csv", 14);
    
    ReadFromCSV(semM, fpath+"sem15.csv", 15);
    ReadFromCSV(semM, fpath+"sem16.csv", 16);
    ReadFromCSV(semM, fpath+"sem17.csv", 17);
    ReadFromCSV(semM, fpath+"sem18.csv", 18);
    ReadFromCSV(semM, fpath+"sem19.csv", 19);
    
    ReadFromCSV(semM, fpath+"sem20.csv", 20);
    
    ReadFromCSV(instI, fpath+"instance.csv", 0);
}


void testSegmentGrow(std::string fpath){
    CDataTempl<UINT32> shape(2);
    ReadFromCSV(shape, fpath+"size.csv", 0);
    UINT32 imgHt = shape.GetData(0);
    UINT32 imgWd = shape.GetData(1);
    
    CDataTempl<float> distM(imgHt, imgWd, 4);
    CDataTempl<float> semM(imgHt, imgWd, 21);
    CDataTempl<UINT32> instI(imgHt, imgWd);
    PrepareData(fpath, distM, semM, instI);
#ifdef DEBUG_FINAL_TRIMAP
    CDataTempl<float> maskI;
#else
    CDataTempl<UINT32> maskI;
#endif
    ProposalGenerate(distM, semM, instI, maskI);
    
    // visulizing.
    cout<<"channel number is: "<<maskI.GetZDim()<<endl;
    return; 
    string py_command = "python pyShow.py";
    for(UINT32 k=0; k < maskI.GetZDim(); k++){
        WriteToCSV(maskI, "./output/test.csv", k);
        system(py_command.c_str());
    } 
}


int main(){
    UINT32 cnt = 0;
    std::string fpath;
    
    cout<<endl<<"*** Please input fname (starting with 2): No."<<cnt<<endl;
    cin >> fpath;
    while(fpath[0] == '2'){
        cout<<"Processing image: "<<fpath<<endl;
        fpath = "./input/"+fpath+"/";
        testSegmentGrow(fpath);
        cnt += 1; 
        cout<<endl<<"*** Please input fname (starting with 2): No."<<cnt<<endl;
        cin >> fpath;
    }
    return 0;
}
    

