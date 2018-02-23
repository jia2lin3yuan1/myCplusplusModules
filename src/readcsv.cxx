#include <fstream>
#include <sstream>
#include <iostream>
int main()
{
    float data[500][373];
    std::ifstream file("dist0.csv");

    for(int row = 0; row < 500; ++row)
    {
        std::string line;
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);

        for (int col = 0; col < 373; ++col)
        {
            std::string val;
            std::getline(iss, val, ',');
            if ( !iss.good() )
                break;

            std::stringstream convertor(val);
            convertor >> data[row][col];

            if(row == 251){
                std::cout<<data[row][col]<<", ";
            }
        }
    }

    std::cout<<std::endl;
    return 0;
}
