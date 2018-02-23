#include <fstream>
#include <sstream>
#include <iostream>
int main()
{
    // read
    float data[500][375];
    std::ifstream file("dist0.csv");

    std::string line;
    for(int row = 0; row < 500; ++row)
    {
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::stringstream iss(line);

        std::string val;
        for (int col = 0; col < 375; ++col)
        {
            std::getline(iss, val, ',');
            //if ( !iss.good() )
            //    break;

            std::stringstream convertor(val);
            convertor >> data[row][col];

            
            if(row == 400){
                std::cout<<col<<'-'<<data[row][col]<<", ";
            }
        }
    }
    std::cout<<std::endl;
    file.close();

    // write.
    std::ofstream outfile("dist0_r.csv");
    for(int row = 0; row < 500; ++row){
        outfile<<data[row][0];
        for (int col = 1; col < 375; ++col){
            outfile<<","<<data[row][col];
        }
        outfile<<std::endl;
    }

    outfile.close();

    return 0;
}
