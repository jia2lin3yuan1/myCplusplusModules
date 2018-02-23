#include "DataTemplate.hxx"

void testDataAccess(){
   
    std::vector<int> arr(12, 5);
    CDataTempl<int> myData(3,4);
    myData.Assign(arr);// output 5, 5, 5
    std::cout << "value is: "<<myData.GetData(-1,1) << ", "<<myData.GetDataByIdx(9)<<", "<< myData.GetData(5,-2) << std::endl;
    std::cout<<"GT: 5,5,5"<<std::endl;
    
    for(int k=0; k < 12; k++){
        arr[k] = k;
    }
    
    myData.Assign(arr);// output:: 1, 9, 8
    std::cout << "value is: "<<myData.GetData(-1,1) << ", "<<myData.GetDataByIdx(9)<<", "<< myData.GetData(5,-2) << std::endl;
    std::cout<<"GT: 1,9,8"<<std::endl;

    CDataTempl<int> row(4);
    myData.GetRow(row, 1);
    std::cout<<"Row value is:"; // 4,5,6,7
    for(int k=0; k < 4; k++){
        std::cout<<row.GetData(k)<<", ";
    }
    std::cout<<std::endl<<"GT: 4,5,6, 7"<<std::endl;

    CDataTempl<int> col(3);
    myData.GetColumn(col, 2); // 2,6,10
    std::cout<<std::endl<<"Col value is:";
    for(int k=0; k < 3; k++){
        std::cout<<col.GetData(k)<<", ";
    }
    std::cout<<std::endl<<"GT: 2,6,10"<<std::endl;
    
}

int main(){
    testDataAccess();
    return 0;
}
