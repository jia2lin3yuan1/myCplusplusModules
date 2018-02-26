
#include "utils/graph.hxx"
#include "utils/read_write_img.hxx"

void PrepareData(std::string fpath,  CDataTempl<UINT32> &labelI){
    ReadFromCSV(labelI, fpath+"label.csv");
}

void testGraph(std::string fpath, UINT32 imgHt, UINT32 imgWd){

    CDataTempl<UINT32> labelI(imgHt, imgWd);
    PrepareData(fpath, labelI);
    
    Graph<Supix, Edge, BndryPix> mygraph(imgHt, imgWd);
    mygraph.AssignInputLabel(&labelI);
    mygraph.CreateGraphFromLabelI();

    mygraph.MergeSuperPixels(1, 2);


    CDataTempl<UINT32> outLabel;
    outLabel = mygraph.AssignOutputLabel();
    
    WriteToCSV(outLabel, "./output/test_graph.csv");
    string py_command = "python pyShow.py";
    system(py_command.c_str());
}

int main(){
    UINT32 imgHt = 375;
    UINT32 imgWd = 500;
    std::string fpath = "./input/"; 
    
    testGraph(fpath, imgHt, imgWd);

    return 0;
}


