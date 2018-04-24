#include <iostream>
#include <queue>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring> // for memset, memcpy

using namespace std;

struct compare
{
  bool operator()(pair<int, float*>& l, pair<int, float*>& r)
  {
      return *(l.second) > *(r.second);
  }
};


struct A{
    int a;
    float b;
    vector<double> c;
    A(){
        a = 5;
        b = 9.2;

        c.resize(6, 2);
        c[1] = 1;
        c[3] = 3;
        c[5] = 5;
    }
};
string printMap(auto *pMap){
    for(auto ele:*pMap){
        cout << ele.second<<" -- ";
    }
    cout<<endl;
    auto it = pMap->begin();
    cout<< (++it)->second<<endl;

    return it->second; 
}

int main(){

    vector<int>  my_vec;
    my_vec.push_back(1);
    my_vec.push_back(7);
    my_vec.push_back(6);
    my_vec.push_back(5);
    my_vec.push_back(4);
    my_vec.push_back(3);
    my_vec.push_back(2);
    cout<<"output vector:: "<< endl;
    for(auto ele : my_vec){
        cout<<ele<<", ";
    }
    cout<<endl;

    my_vec.assign(18, 30);
    cout<<"output vector:: "<< endl;
    for(auto ele : my_vec){
        cout<<ele<<", ";
    }
    cout<<endl;


    return 0;

}
