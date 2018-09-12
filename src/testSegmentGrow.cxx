
#include "ProposalGenerate.hxx"
#include "utils/read_write_img.hxx"

/*
 * Before test this case, pass the test of segment fitting first.
 *
 */


void PrepareData(std::string fpath, CDataTempl<double> &distM, CDataTempl<double> &semM, CDataTempl<int> &instI){
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


void testSegmentGrow(std::string fpath, std::string fname){
    CDataTempl<int> shape(2);
    ReadFromCSV(shape, fpath+"size.csv", 0);
    int imgHt = shape.GetData(0);
    int imgWd = shape.GetData(1);
    
    CDataTempl<double> distM(imgHt, imgWd, 4);
    CDataTempl<double> semM(imgHt, imgWd, 21);
    CDataTempl<int> instI(imgHt, imgWd);
    PrepareData(fpath, distM, semM, instI);
#ifdef DEBUG_FINAL_TRIMAP
    CDataTempl<double> maskI;
#else
    CDataTempl<int> maskI;
#endif
    ProposalGenerate(distM, semM, instI, maskI);
    
    // visulizing.
    cout<<"channel number is: "<<maskI.GetZDim()<<endl;
    //return; 
    for(int k=0; k < maskI.GetZDim(); k++){
        string outPath   = "/media/yuanjial/LargeDrive/Results/python-instanceinference/Cython_output/"+fname+std::string("_")+std::to_string(k)+".png";
        string py_command = std::string("python pyShow.py") + std::string(" --o ") + outPath;
        //string py_command = std::string("python pyShow.py");
        cout<<py_command<<endl; 
        WriteToCSV(maskI, "./output/test.csv", k);
        system(py_command.c_str());
    } 
}


int main(){
    int cnt = 0;
    std::string fname;
    
    std::string fdir;
    cout<<endl<<"*** Please the folder name of input images."<<cnt<<endl;
    cin >> fdir;
    cout<<endl<<"*** Please input fname (starting with 2): No."<<cnt<<endl;
    cin >> fname;
    while(fname[0] == '2'){
        cout<<"Processing image: "<<fname<<endl;
        std::string fpath = "./" + fdir + "/" + fname + "/";
        testSegmentGrow(fpath, fname);
        cnt += 1; 
        cout<<endl<<"*** Please input fname (starting with 2): No."<<cnt<<endl;
        cin >> fname;
    }
    return 0;
}
    

