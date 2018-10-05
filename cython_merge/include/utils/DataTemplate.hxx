#ifndef _DATA_TEMPLATE_HXX
#define _DATA_TEMPLATE_HXX

#include <cassert>
#include <cstring>
#include <algorithm>

#include "PlatformDefs.hxx"

template <typename BT>
class CDataTempl{ // 2D matrix.
protected:
    BT   *m_pBuf = NULL; // The data is arraged as: [[[z0y0x0, z0y0x1, ...,z0y0x_n], [z0y1x0, z0y1x1, ..., z0y1x_n],.....[...., z0y_mx_n]]
                         //                          [[z1y0x0, z1y0x1, ...,z1y0x_n], [z1y1x0, z1y1x1, ..., z1y1x_n],.....[...., z1y_mx_n]]
                         //                          [[...........................]..[............................]......[..............]] ]
    int m_xDim, m_yDim, m_zDim, m_size;

    void AllocateBuf(){
        Destroy();
        m_pBuf = new BT[m_size];
        memset(m_pBuf, 0, m_size*sizeof(BT));
    }
    void Destroy(){
        if(m_pBuf){
            delete [] m_pBuf;
            m_pBuf = NULL;
        }
    }

public:
    CDataTempl(){
        m_xDim = m_yDim = m_zDim = m_size = 0;
    }
    CDataTempl(int dy, int dx=1, int dz=1){
        this->Init(dy, dx, dz);
    }
    ~CDataTempl(){
        Destroy();
        m_xDim = m_yDim = m_zDim = m_size = 0;
    }

    void Init(int dy, int dx=1, int dz=1){
        m_xDim = dx; m_yDim = dy; m_zDim = dz;
        m_size = m_xDim*m_yDim*m_zDim;
        this->AllocateBuf();
    }

    CDataTempl<BT>& operator=(CDataTempl<BT>& data){
        Init(data.GetYDim(), data.GetXDim(), data.GetZDim());
        Copy(data);
        return *this;
    }
   
    BT operator()(int y, int x=0, int z=0) const{
        y = y<0? 0 : (y> m_yDim-1? m_yDim-1 : y);
        x = x<0? 0 : (x> m_xDim-1? m_xDim-1 : x);
        z = z<0? 0 : (z> m_zDim-1? m_zDim-1 : z);
        return m_pBuf[this->Coordinate2Index(y,x,z)];
    }

    // inline function.
    int GetXDim() const;
    int GetYDim() const;
    int GetZDim() const;
    int GetSize() const;
    int Coordinate2Index(int y, int x=0, int z=0) const;
   
    // assign value
    void SetData(BT val, int y, int x=0, int z=0);
    void SetDataByIdx(BT val, int k);
    void ResetDataFromVector(std::vector<int> idxs, BT val, int z=0);
    void ResetBulkData(BT val, int y0, int ys, int x0=0, int xs=1, int z0=0, int zs =1);
    void Reset(BT val);
    
    void Copy(CDataTempl<BT> &data);
    void AssignFromVector(const std::vector<BT> &dataV);
    void AssignFromVector(BT* dataV);
    void Minimum(const CDataTempl<BT> &data0, const CDataTempl<BT> &data1);
    void Add(const CDataTempl<BT> &data0, const CDataTempl<BT> &data1);
    
    // fetch data..
    BT GetData(SINT32 y, SINT32 x=0, SINT32 zs=0) const;
    BT GetDataByIdx(int k) const;
    void GetBulkData(std::vector<BT> &pixV, int y0, int ys, int x0=0, int xs=1, int z0=0, int zs=1);
    void GetRow(CDataTempl<BT> &row, int y, int z=0);
    void GetColumn(CDataTempl<BT> &col, int x, int z=0);

    // math on CDataTemplate.
    void argmax(CDataTempl<int> &indexI, int dim);
    template<typename T2>
    void Mask(CDataTempl<T2> &boolI, BT val);
    double Mean();
    double BulkMean(int y0, int ys, int x0=0, int xs=1, int z0=0, int zs=1);
    BT Max();
    BT BulkMax(int y0, int ys, int x0=0, int xs=1, int z0=0, int zs=1);

    // functions only work on Zdim.
    void MeanZ(const std::vector<int> &in_yxIds, std::vector<BT> &outV);
    template<typename T2>
    void MeanZ(const CDataTempl<T2> &maskI, T2 val, std::vector<BT> &outV);
    
    // serve for SegmentGrow, only work when m_zDim=1
    void FindBoundaryOnMask(std::vector<std::pair<int, int> > &bds, BT fgV);
    int FindMaskBorderPoint(int py, int px, int st, int end, int step=1);
    int ReplaceByValue(BT src, BT dst);
    template<typename T2>
    void ModifyMaskOnNonZeros(const CDataTempl<T2> &matA, BT val);
};
    // inline function.
template <typename BT>
int CDataTempl<BT>::GetXDim() const {return m_xDim;}
template <typename BT>
int CDataTempl<BT>::GetYDim() const {return m_yDim;}
template <typename BT>
int CDataTempl<BT>::GetZDim() const {return m_zDim;}
template <typename BT>
int CDataTempl<BT>::GetSize() const {return m_size;}
template <typename BT>
int CDataTempl<BT>::Coordinate2Index(int y, int x, int z) const {return (z*m_yDim+y)*m_xDim + x;}

// assign and modify data.
template <typename BT>
void CDataTempl<BT>::SetData(BT val, int y, int x, int z){
    assert(y>=0 && y<m_yDim && x>=0 && x<m_xDim && z>=0 && z<m_zDim);
    m_pBuf[this->Coordinate2Index(y,x, z)] = val;
}
template <typename BT>
void CDataTempl<BT>::SetDataByIdx(BT val, int k){
    assert(k>=0 && k<m_size);
    m_pBuf[k] = val;
}
template <typename BT>
void CDataTempl<BT>::ResetDataFromVector(std::vector<int> idxs, BT val, int z){
    int base_k = z * m_yDim * m_xDim;
    for(SINT32 k= idxs.size()-1; k >= 0; k--){
        m_pBuf[idxs[k]+base_k] = val;
    }
}
template <typename BT>
void CDataTempl<BT>::ResetBulkData(BT val, int y0, int ys, int x0, int xs, int z0, int zs){
    int y1 = y0 + ys;
    int x1 = x0 + xs;
    int z1 = z0 + zs;
    for(int z=z0; z<z1; z++)    
        for(int y=y0; y<y1; y++)    
            for(int x=x0; x<x1; x++)
                this->SetData(val, y, x, z);
}

template <typename BT>
void CDataTempl<BT>::Reset(BT val){
    for(int k=0; k<m_size; k++){
        m_pBuf[k] = val;
    }
}

template <typename BT>
void CDataTempl<BT>::Copy(CDataTempl<BT> &data){
    assert(m_xDim==data.GetXDim() && m_yDim==data.GetYDim() && m_zDim==data.GetZDim());
    for(int k=0; k<m_size; k++){
        m_pBuf[k] = data.GetDataByIdx(k);
    }
}
template <typename BT>
void CDataTempl<BT>::AssignFromVector(BT* dataV){
    assert(m_size == dataV.size());
    for(int k=0; k<m_size; k++){
        m_pBuf[k] = dataV[k];
    }
}
template <typename BT>
void CDataTempl<BT>::AssignFromVector(const std::vector<BT> &dataV){
    assert(m_size == dataV.size());
    for(int k=0; k<m_size; k++){
        m_pBuf[k] = dataV[k];
    }
}
template <typename BT>
void CDataTempl<BT>::Minimum(const CDataTempl<BT> &data0, const CDataTempl<BT> &data1){
    assert(data0.GetXDim()==data1.GetXDim() && data0.GetYDim()==data1.GetYDim() && data0.GetZDim()==data1.GetZDim());
    assert(m_xDim==data1.GetXDim() && m_yDim==data1.GetYDim() && m_zDim==data1.GetZDim());
    for(int k=0; k<m_size; k++){
        m_pBuf[k] = min(data0.GetDataByIdx(k) + data1.GetDataByIdx(k));
    }
}
template <typename BT>
void CDataTempl<BT>::Add(const CDataTempl<BT> &data0, const CDataTempl<BT> &data1){
    assert(data0.GetXDim()==data1.GetXDim() && data0.GetYDim()==data1.GetYDim() && data0.GetZDim()==data1.GetZDim());
    assert(m_xDim==data1.GetXDim() && m_yDim==data1.GetYDim() && m_zDim==data1.GetZDim());
    for(int k=0; k<m_size; k++){
        m_pBuf[k] = data0.GetDataByIdx(k) + data1.GetDataByIdx(k);
    }
}
    
// fetching data.
template <typename BT>
BT CDataTempl<BT>::GetData(SINT32 y, SINT32 x, SINT32 z) const {
    y = y<0? 0 : (y> SINT32(m_yDim)-1? SINT32(m_yDim)-1 : y);
    x = x<0? 0 : (x> SINT32(m_xDim)-1? SINT32(m_xDim)-1 : x);
    z = z<0? 0 : (z> SINT32(m_zDim)-1? SINT32(m_zDim)-1 : z);
    return m_pBuf[this->Coordinate2Index(y,x,z)];
}
template <typename BT>
BT CDataTempl<BT>::GetDataByIdx(int k) const{ 
    assert(k>=0 && k< m_size);    
    return m_pBuf[k];
}
template <typename BT>
void CDataTempl<BT>::GetBulkData(std::vector<BT> &pixV, int y0, int ys, int x0, int xs, int z0, int zs){ 
    int y1 = y0 + ys;
    int x1 = x0 + xs;
    int z1 = z0 + zs;
    assert(x0>=0 && x1<m_xDim && y0>=0 && y1<m_yDim && z0>=0 && z1<m_zDim);

    int cnt = 0;
    for(int z=z0; z < z1; z++){
        for(int y=y0; y < y1; y++){
            for(int x=x0; x < x1; x++){
                pixV[cnt] = this->GetData(y, x, z);
                cnt += 1;
            }
        }
    }
}
template <typename BT>
void CDataTempl<BT>::GetRow(CDataTempl<BT> &row, int y, int z){
    assert(row.GetSize() == m_xDim);
    int idx = ((z*m_yDim) + y)*m_xDim + 0;
    for(int k =0; k < m_xDim; k ++){
        row.SetData(m_pBuf[idx], k);
        idx += 1;
    }
}
template <typename BT>
void CDataTempl<BT>::GetColumn(CDataTempl<BT> &col, int x, int z){
    assert(col.GetSize() == m_yDim);
    int idx = ((z*m_yDim) + 0)*m_xDim + x;
    for(int k =0; k < m_yDim; k ++){
        col.SetData(m_pBuf[idx], k);
        idx += m_xDim;
    }
}

// math on CDataTemplate.
template <typename BT>
void CDataTempl<BT>::argmax(CDataTempl<int> &indexI, int dim){
    int loop_len = dim==0? m_yDim*m_zDim : (dim==1? m_zDim : 1);
    int step_len = dim==0? 1 : (dim==1? m_xDim : m_yDim*m_xDim);
    int  cmp_len = dim==0? m_xDim : (dim==1? m_yDim : m_zDim);
    assert(indexI.GetSize() == loop_len*step_len);
    
    for(int k=0; k < loop_len; k++){
        int idx = k*step_len*cmp_len;
        for(int m=0; m < step_len; m++){
            BT maxV = m_pBuf[idx+m], maxIdx = 0;
            for(int i=1; i<cmp_len; i++){
                if(m_pBuf[idx+m+i*step_len] > maxV){
                    maxV   = m_pBuf[idx+m+i*step_len];
                    maxIdx = i;
                }
            }
            indexI.SetDataByIdx(maxIdx, k*loop_len+m);
        } 
    }
}

template <typename BT>
template <typename T2>
void CDataTempl<BT>::Mask(CDataTempl<T2> &boolI, BT val){
    assert(m_size == boolI.GetSize());
    for(int k=0; k<m_size; k++){
        if(m_pBuf[k] == val)
            boolI.SetDataByIdx(1, k);
        else
            boolI.SetDataByIdx(0, k);
    }
}

template <typename BT>
double CDataTempl<BT>::Mean(){
    return this->BulkMean(0, m_yDim, 0, m_xDim, 0, m_zDim);
}
template <typename BT>
double CDataTempl<BT>::BulkMean(int y0, int ys, int x0, int xs, int z0, int zs){
    int size = zs*ys*xs;
    assert(size>0);
    int y1 = y0 + ys;
    int x1 = x0 + xs;
    int z1 = z0 + zs;

    BT sumV = this->GetData(y0, x0, z0);
    for(int z=z0; z<z1; z++)
        for(int y=y0; y<y1; y++)
            for(int x=x0; x<x1; x++)
                sumV += this->GetData(y,x,z);
    return double(sumV)/double(size);
}
template <typename BT>
BT CDataTempl<BT>::Max(){
    return this->BulkMax(0, m_yDim, 0, m_xDim, 0, m_zDim);
}
template <typename BT>
BT CDataTempl<BT>::BulkMax(int y0, int ys, int x0, int xs, int z0, int zs){
    int y1 = y0 + ys;
    int x1 = x0 + xs;
    int z1 = y0 + zs;
    
    BT maxV = this->GetData(y0, x0, z0);
    for(int z=z0; z<z1; z++)
        for(int y=y0; y<y1; y++)
            for(int x=x0; x<x1; x++)
                maxV = maxV < this->GetData(y,x,z)? this->GetData(y,x,z) : maxV;
    return maxV;
}
// functions only work on Zdim.
template <typename BT>
void CDataTempl<BT>::MeanZ(const std::vector<int> &in_yxIds, std::vector<BT> &outV){
    for(int z=0; z<m_zDim; z++){
        outV[z] = 0;
        int base_k = z*m_yDim*m_xDim;
        for(int k = 0; k < in_yxIds.size(); k++ ) {
            outV[z] += this->GetDataByIdx(base_k + in_yxIds[k]); 
        }

        outV[z] = outV[z]/in_yxIds.size();
    }
}

template <typename BT>
template <typename T2>
void CDataTempl<BT>::MeanZ(const CDataTempl<T2> &maskI, T2 val, std::vector<BT> &outV){
    int sizeI = maskI.GetSize();
    assert(sizeI == m_xDim*m_yDim);

    for(int z=0; z<m_zDim; z++){
        outV[z] = 0;
        int cnt  = 0;
        int base_k = z*m_yDim*m_xDim;
        for(int k = 0; k < sizeI; k++ ) {
            if(maskI.GetDataByIdx(k) == val){
                cnt     += 1;
                outV[z] += this->GetDataByIdx(base_k + k); 
            }
        }
        outV[z] = outV[z]/cnt;
    }
}

// serve for SegmentGrow.
template <typename BT>
void CDataTempl<BT>::FindBoundaryOnMask(std::vector<std::pair<int, int> > &bds, BT fgV){
    assert(m_zDim==1);
    CDataTempl<BT> visitedMask(m_yDim, m_xDim, m_zDim); 
    
    // left to right, up to down, find first fg pixel.
    std::stack<std::pair<int, int> > ptStack;
    for(int k=0; k < m_size; k++){
        if(this->GetDataByIdx(k) == fgV){
            visitedMask.SetDataByIdx(1, k);
            ptStack.push(std::make_pair(k/m_xDim, k%m_xDim));
            break;
        }
    }

    // find all bd pixels along boundary.
    int y=0, x=0, cnt=0;
    int nei[18];
    while(!ptStack.empty()){
        y   = ptStack.top().first;
        x   = ptStack.top().second;
        ptStack.pop();
        cnt  = 0;
        for(SINT32 sy=-1; sy<=1; sy++){
            if(y+sy <0 || y+sy >= m_yDim)
                continue;
            for(SINT32 sx=-1; sx<=1; sx++){
                if(x+sx <0 || x+sx >= m_xDim)
                    continue;
                if(this->GetData(y+sy, x+sx)==fgV){
                    nei[cnt]    = y+sy;
                    nei[cnt +1] = x+sx;
                    cnt += 2;
                }
            }
        }
        
        if(cnt < 18){
            bds.push_back(std::make_pair(y, x));
            for(int k = 0; k < cnt; k+= 2){
                if(visitedMask.GetData(nei[k], nei[k+1]) == 0){
                    ptStack.push(std::make_pair(nei[k], nei[k+1]));
                    visitedMask.SetData(1, nei[k], nei[k+1]);
                }
            }
        }
    }
}



template <typename BT>
int CDataTempl<BT>::FindMaskBorderPoint(int py, int px, int st, int end, int step){
    assert(m_zDim == 1);
    int idx = this->Coordinate2Index(py, px);
   
    int ptK = step == 1? px : py;
    if(ptK >0 && m_pBuf[idx-step]>0){
        while(ptK > st && m_pBuf[idx-step]>0){
            ptK -= 1;
            idx -= step;
        } 
    }
    else{
        while(ptK < end-1 && m_pBuf[idx+step] > 0){
            ptK += 1;
            idx += step;
        }
    }
    return ptK;
}

template <typename BT>
int CDataTempl<BT>::ReplaceByValue(BT src, BT dst){
    assert(m_zDim == 1);
    int cnt = 0;
    for(int k=0; k < m_size; k++){
        if(m_pBuf[k] == src){
            cnt += 1;
            m_pBuf[k] = dst;
        }
    }
    return cnt;
}
template <typename BT>
template<typename T2>
void CDataTempl<BT>::ModifyMaskOnNonZeros(const CDataTempl<T2> &matA, BT val){
    assert(m_zDim == 1);
    assert(m_xDim==matA.GetXDim() && m_yDim==matA.GetYDim() && m_zDim==matA.GetZDim());
    for(int k=0; k < m_size; k++){
        if(matA.GetDataByIdx(k) != 0){
            m_pBuf[k] = val;
        }
    }
}

/*
 */


#endif
