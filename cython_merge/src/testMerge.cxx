#include "MergeMain.hxx"
#include "utils/read_write_img.hxx"

/*
 * Before test this case, pass the test of segment fitting first.
 *
 */


void PrepareData(std::string fpath, CDataTempl<double> &predM, CDataTempl<int> &oversegI){
    ReadFromCSV(predM, fpath+"pred.csv", 0);
    ReadFromCSV(oversegI, fpath+"overseg.csv", 0);
}


void testSegmentGrow(std::string fpath, std::string fname){
    CDataTempl<int> shape(4);
    ReadFromCSV(shape, fpath+"size.csv", 0);
    int imgHt   = shape.GetData(0);
    int imgWd   = shape.GetData(1);
    int num_obj = shape.GetData(2);
    double obj_cen_maxV = shape.GetData(3);
    
    CDataTempl<double> predM(imgHt, imgWd);
    CDataTempl<int>    oversegI(imgHt, imgWd);
    PrepareData(fpath, predM, oversegI);

    CDataTempl<int> maskI;
    ProposalGenerate(num_obj, obj_cen_maxV, predM, oversegI, maskI);
    
    // visulizing.
    cout<<"channel number is: "<<maskI.GetZDim()<<endl;
    return; 
    
    
    for(int k=0; k < maskI.GetZDim(); k++){
        string outPath   = "/media/yuanjial/LargeDrive/Results/python-instanceinference/Cython_output/"+fname+std::string("_")+std::to_string(k)+".png";
        // string py_command = std::string("python pyShow.py") + std::string(" --o ") + outPath;
        string py_command = std::string("python pyShow.py");
        cout<<py_command<<endl; 
        WriteToCSV(maskI, "./output/test.csv", k);
        system(py_command.c_str());
    } 
}


int main(){
    int cnt = 0;
    std::string fname;
    
    std::string fdir = "input_cls_test";
    /*
    cout<<endl<<"*** Please the folder name of input images."<<cnt<<endl;
    cin >> fdir;
    cout<<endl<<"*** Please input fname (starting with 2): No."<<cnt<<endl;
    cin >> fname;
    */
    fname = "2007_001239";
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
    

