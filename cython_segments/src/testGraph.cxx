
#include "utils/graph.hxx"
#include "utils/read_write_img.hxx"

void PrepareData(std::string fpath,  CDataTempl<int> &labelI){
    ReadFromCSV(labelI, fpath+"label.csv");
}

void testGraph(std::string fpath, int imgHt, int imgWd){

    CDataTempl<int> labelI(imgHt, imgWd);
    PrepareData(fpath, labelI);
    
    Graph<Supix, Edge, BndryPix> mygraph(imgHt, imgWd);
    mygraph.AssignInputLabel(&labelI);
    mygraph.CreateGraphFromLabelI();

    mygraph.MergeSuperPixels(1, 2);


    CDataTempl<int> outLabel;
    outLabel = mygraph.AssignOutputLabel();
    
    WriteToCSV(outLabel, "./output/test_graph.csv");
    string py_command = "python pyShow.py";
    system(py_command.c_str());
}

int main(){
    int imgHt = 375;
    int imgWd = 500;
    std::string fpath = "./input/"; 
    
    testGraph(fpath, imgHt, imgWd);

    return 0;
}


