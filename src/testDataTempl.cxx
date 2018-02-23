#include "DataTemplate.hxx"

int main(){
   
    std::vector<int> arr(12, 5);
    CDataTempl<int> myData(3,4);
    myData.Assign(arr);
    std::cout << "value is: "<<myData.GetData(1,1) << ", "<< myData.GetData(5,2) << std::endl;
    
    for(int k=0; k < 12; k++){
        arr[k] = k;
    }
    
    myData.Assign(arr);
    std::cout << "After Reset: value is: "<<myData.GetData(5,5) << ", "<< myData.GetData(-1, 3) << std::endl;

    CDataTempl<int> row(4);
    myData.GetRow(row, 1);
    std::cout<<"Row value is:";
    for(int k=0; k < 4; k++){
        std::cout<<row.GetData(k)<<", ";
    }
    std::cout<<"DOne"<<std::endl;

    CDataTempl<int> col(3);
    myData.GetColumn(col, 2);
    std::cout<<std::endl<<"Col value is:";
    for(int k=0; k < 3; k++){
        std::cout<<col.GetData(k)<<", ";
    }
    std::cout<<std::endl;
    
    return 0;
}
