#ifndef _READ_WRITE_IMG_HXX
#define _READ_WRITE_IMG_HXX


#include <fstream>
#include <sstream>
#include <iostream>

#include "DataTemplate.hxx"
/*
 *The functions in this file are used to read/write 2D matrix from CSV file. The 2D matrix could be saved / readed by Python. 
 * np.savetxt('fname.csv', rgbI[...,0], delimiter=',', fmt='%1.5e')
 * rI = np.loadtxt('fname.csv', delimiter=',')
 */

template<typename DT>
void ReadFromCSV(CDataTempl<DT> &img, std::string fpath,int imgCh =0)
{
    UINT32 imgHt = img.GetYDim();
    UINT32 imgWd = img.GetXDim();

    float rdVal;
    std::string line;
    std::ifstream file(fpath);
    for(int row = 0; row < imgHt; ++row)
    {
        std::getline(file, line);
        if ( !file.good() )
            break;

        std::string val;
        std::stringstream iss(line);
        for (int col = 0; col < imgWd; ++col)
        {
            std::getline(iss, val, ',');
            //if ( !iss.good() )
            //    break;

            std::stringstream convertor(val);
            convertor >> rdVal;
            
            img.SetData(DT(rdVal), row, col, imgCh);
        }
    }
}


template<typename DT>
void WriteToCSV(CDataTempl<DT> &img, std::string fpath, int imgCh=0){
    UINT32 imgHt = img.GetYDim();
    UINT32 imgWd = img.GetXDim();

    std::ofstream outfile(fpath);
    
    for(int row = 0; row < imgHt; ++row){
        outfile<< img.GetData(row, 0, imgCh);
        
        for(int col = 1; col < imgWd; ++col){
            outfile<<","<<img.GetData(row, col, imgCh);
        }
        outfile<<std::endl;
    }

    outfile.close();
}
 #endif
