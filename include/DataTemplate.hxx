#ifndef _DATA_TEMPLATE_HXX
#define _DATA_TEMPLATE_HXX

#include <cassert>
#include <cstring>
#include <algorithm>

#include "PlatformDefs.hxx"

template <typename BT>
class CDataTempl{
protected:
    BT *m_pBuf=NULL; // The data is arraged as: [x0y0z0, x1y0z0, ...,xNy0z0, x0y1,z0, x1y1z0, ..., xNy1z0,........., XNYMz0, 
                //                          x0y0z1, x1y0z1, ..., ....]
    UINT32 m_xDim, m_yDim, m_zDim, m_size;

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
        m_xDim = m_yDim = m_zDim = 0;
        m_size = m_xDim*m_yDim*m_zDim;
        m_pBuf = NULL;
    }
    CDataTempl(UINT32 dy, UINT32 dx=1, UINT32 dz=1){
        this->Init(dy, dx, dz);
    }
    ~CDataTempl(){
        Destroy();
        m_xDim = m_yDim = m_zDim = 0;
    }

    void Init(UINT32 dy, UINT32 dx=1, UINT32 dz=1){
        m_size = dx*dy*dz;
        m_xDim = dx; m_yDim = dy; m_zDim = dz;
        this->AllocateBuf();
    }

    CDataTempl<BT>& operator=(CDataTempl<BT>& data){
        Init(data.GetYDim(), data.GetXDim(), data.GetZDim());
        Copy(data);
        return *this;
    }

    CDataTempl<BT> operator+(CDataTempl<BT> &data){
        assert(m_xDim==data.GetXDim() && m_yDim==data.GetYDim() && m_zDim==data.GetZDim());
        CDataTempl<BT> outData(m_yDim, m_xDim, m_zDim);
        for(UINT32 k=0; k<m_size; k++){
            BT val = this->GetDataByIdx(k) + data.GetDataByIdx(k);
            outData.SetDataByIdx(val, k);
        }

        return outData;
    }


    // inline function.
    UINT32 GetXDim() const;
    UINT32 GetYDim() const;
    UINT32 GetZDim() const;
    BT GetDataByIdx(UINT32 k) const;
    UINT32 Coordinate2Index(UINT32 y, UINT32 x, UINT32 z) const;
    
    void SetData(BT val, UINT32 y, UINT32 x=0, UINT32 z=0);
    void SetDataByIdx(BT val, UINT32 k);

    // normal member functions.
    BT GetData(SINT32 y, SINT32 x=0, SINT32 z=0);
    void GetBulkData(BT data[], UINT32 sy, UINT32 dy, UINT32 sx=0, UINT32 dx=0, UINT32 sz=0, UINT32 dz=0);
    
    void Copy(CDataTempl<BT> &data);
    void Assign(std::vector<BT> &data_vec);
    void ResetBulkData(BT val, UINT32 y0, UINT32 ys, UINT32 x0=0, UINT32 xs=1, UINT32 z0=0, UINT32 zs=1);
    void Reset(BT initVal);
    
    void GetRow(CDataTempl<BT> &row, UINT32 y, UINT32 z=0);
    void GetColumn(CDataTempl<BT> &col, UINT32 x, UINT32 z=0);

    // math on CDataTemplate.
    BT Mean();
    BT Max();
    BT BulkMax(UINT32 y0, UINT32 ys, UINT32 x0=0, UINT32 xs=1, UINT32 z0=0, UINT32 zs=1);
    void Minimum(CDataTempl<BT> &data0, CDataTempl<BT> &data1);
    BT BulkMean(UINT32 y0, UINT32 ys, UINT32 x0=0, UINT32 xs=1, UINT32 z0=0, UINT32 zs=1);

    // serve for SegmentGrow.
    void FindBoundaryOnMask(std::vector<std::pair<UINT32, UINT32> > &bds, BT fgV=1);
    UINT32 FindMaskBorderPoint(UINT32 py, UINT32 px, UINT32 st, UINT32 end,  UINT32 step=1);
    UINT32 ReplaceByValue(BT srcV, BT dstV);
    void ModifyMaskOnNonZeros(CDataTempl<BT> &matA, BT val);
};


template <typename BT>
UINT32 CDataTempl<BT>::ReplaceByValue(BT srcV, BT dstV){
    UINT32 cnt = 0;
    for(UINT32 k=0; k < m_size; k++){
        if(m_pBuf[k] == srcV){
            cnt += 1;
            m_pBuf[k] = dstV;
        }
    }
    return cnt;
}

template <typename BT>
void CDataTempl<BT>::ModifyMaskOnNonZeros(CDataTempl<BT> &matA, BT val){
    assert(m_xDim==matA.GetXDim() && m_yDim==matA.GetYDim() && m_zDim==matA.GetZDim());
    for(UINT32 k=0; k < m_size; k++){
        if(m_pBuf[k] != 0){
            matA.SetDataByIdx(val, k);
        }
    }
}

template <typename BT>
void CDataTempl<BT>::FindBoundaryOnMask(std::vector<std::pair<UINT32, UINT32> > &bds, BT fgV){
    
    std::stack<std::pair<UINT32, UINT32> > ptStack;
    // left to right, up to down, find first fg pixel.
    for(UINT32 y=0; y<m_yDim; y++)
        for(UINT32 x=0; x< m_xDim; x++)
            if(this->GetData(y,x)==fgV){
                ptStack.push(std::make_pair(y, x));
                break;
            }
    
    // find all bd pixels along boundary.
    UINT32 y, x, cnt;
    UINT32 nei[18];
    while(!ptStack.empty()){
        y   = ptStack.top().first;
        x   = ptStack.top().second;
        ptStack.pop();
        cnt = 0;
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
            bds.push_back(std::make_pair(y,x));
            for(int k = 0; k < cnt; k+= 2){
                bds.push_back(std::make_pair(nei[k], nei[k+1]));
            }
        }
    }
}

template <typename BT>
UINT32 CDataTempl<BT>::FindMaskBorderPoint(UINT32 py, UINT32 px, UINT32 st, UINT32 end, UINT32 step){
    assert(m_zDim == 1);
    UINT32 idx = this->Coordinate2Index(py, px);
    UINT32 idx_st  = step>1? this->Coordinate2Index(st, px) : this->Coordinate2Index(py, st); 
    UINT32 idx_end = step>1? this->Coordinate2Index(end, px) : this->Coordinate2Index(py, end); 
    SINT32 cnt = 0;
    if(m_pBuf[idx+step] > 0){
        while(idx <= idx_end && m_pBuf[idx] > 0){
            cnt += 1;
            idx += step;
        }
    }
    else{
        while(idx >= idx_st && m_pBuf[idx] > 0){
            cnt -= 1;
            idx -= step;
        }
    }

    return step>1? py+cnt : px+cnt;
}


template <typename BT>
inline UINT32 CDataTempl<BT>::GetXDim() const {return m_xDim;}
template <typename BT>
inline UINT32 CDataTempl<BT>::GetYDim() const {return m_yDim;}
template <typename BT>
inline UINT32 CDataTempl<BT>::GetZDim() const {return m_zDim;}
template <typename BT>
inline BT CDataTempl<BT>::GetDataByIdx(UINT32 k) const{
    k = _CLIP(k, 0, m_xDim*m_yDim*m_zDim-1);
    return m_pBuf[k];
}
template <typename BT>
inline UINT32 CDataTempl<BT>::Coordinate2Index(UINT32 y, UINT32 x, UINT32 z) const {
    return (z*m_yDim + y)*m_xDim + x;
}

template <typename BT>
inline void CDataTempl<BT>::SetData(BT val, UINT32 y, UINT32 x, UINT32 z){
    m_pBuf[this->Coordinate2Index(y,x,z)] = val;
}
template <typename BT>
inline void CDataTempl<BT>::SetDataByIdx(BT val, UINT32 k){
    m_pBuf[k] = val;
}


// normal member functions.
template <typename BT>
BT CDataTempl<BT>::GetData(SINT32 y, SINT32 x, SINT32 z){
    x = x < 0? 0 : x > m_xDim-1? m_xDim-1 : x;
    y = y < 0? 0 : y > m_yDim-1? m_yDim-1 : y;
    z = z < 0? 0 : z > m_zDim-1? m_zDim-1 : z;
    
    return m_pBuf[this->Coordinate2Index(y,x,z)];
}

template <typename BT>
void CDataTempl<BT>::GetBulkData(BT data[], UINT32 sy, UINT32 dy, UINT32 sx, UINT32 dx, UINT32 sz, UINT32 dz){
   // todo:: 
}

template <typename BT>
void CDataTempl<BT>::Copy(CDataTempl<BT> &data){
    assert(m_xDim==data.GetXDim() && m_yDim==data.GetYDim() && m_zDim==data.GetZDim());
    for(UINT32 k=0; k < m_size; k++){
        m_pBuf[k] = data.GetDataByIdx(k);
    }
}

template <typename BT>
void CDataTempl<BT>::Assign(std::vector<BT> &data_vec){
    assert(m_size == data_vec.size());
    for(UINT32 k=0; k<m_size; k++){
        m_pBuf[k] = data_vec[k];
    }
}

template <typename BT>
void CDataTempl<BT>::ResetBulkData(BT val, UINT32 y0, UINT32 ys, UINT32 x0, UINT32 xs, UINT32 z0, UINT32 zs){
    assert(xs>0 && ys>0 && zs>0);
    UINT32 z1 = z0+zs, y1 = y0+ys, x1 = x0+xs;
    for(UINT32 z=z0; z<z1; z++)
        for(UINT32 y=y0; y<y1; y++)
            for(UINT32 x=x0; x<x1; x++)
                this->SetData(val, y,x,z);
}

template <typename BT>
void CDataTempl<BT>::Reset(BT initVal){
    for(UINT32 k=0; k<m_size; k++)
        m_pBuf[k] = initVal;
}

template <typename BT>
void CDataTempl<BT>::GetRow(CDataTempl<BT> &row, UINT32 y, UINT32 z){
    UINT32 idx = ((z*m_yDim) + y)*m_xDim;
    for(UINT32 k =0; k < m_xDim; k ++){
        row.SetData(m_pBuf[idx], k);
        idx += 1;
    }
}

template <typename BT>
void CDataTempl<BT>::GetColumn(CDataTempl<BT> &col, UINT32 x, UINT32 z){
    UINT32 idx = z*m_yDim*m_xDim + x;
    for(UINT32 k =0; k < m_yDim; k ++){
        col.SetData(m_pBuf[idx], k);
        idx += m_xDim;
    }
}


// math on CDataTemplate.
template <typename BT>
BT CDataTempl<BT>::Mean(){
    return this->BulkMean(0, m_yDim, 0, m_xDim, 0, m_zDim);
}

template <typename BT>
BT CDataTempl<BT>::Max(){
    return BulkMax(0, m_yDim, 0, m_xDim, 0, m_zDim);
}

template <typename BT>
BT CDataTempl<BT>::BulkMax(UINT32 y0, UINT32 ys, UINT32 x0, UINT32 xs, UINT32 z0, UINT32 zs){
    assert(xs>0 && ys>0 && zs>0);
    UINT32 z1 = z0+zs, y1 = y0+ys, x1 = x0+xs;
    
    BT maxV = this->GetData(y0, x0, z0);
    for(UINT32 z=z0; z<z1; z++){
        for(UINT32 y=y0; y<y1; y++){
            for(UINT32 x=x0; x<x1; x++){
                maxV = maxV < this->GetData(y,x,z)? this->GetData(y,x,z):maxV;
            }
        }
    }
    return maxV;
}

template <typename BT>
void CDataTempl<BT>::Minimum(CDataTempl<BT> &data0, CDataTempl<BT> &data1){
    assert(data0.GetXDim()==data1.GetXDim() && data0.GetYDim()==data1.GetYDim() && data0.GetZDim()==data1.GetZDim());
    assert(m_xDim==data1.GetXDim() && m_yDim==data1.GetYDim() && data0.GetZDim()==data1.GetZDim());
    for(UINT32 k=0; k<m_size; k++){
        m_pBuf[k] = std::min(data0.GetDataByIdx(k), data1.GetDataByIdx(k));
    }
}

template <typename BT>
BT CDataTempl<BT>::BulkMean(UINT32 y0, UINT32 ys, UINT32 x0, UINT32 xs, UINT32 z0, UINT32 zs){
    assert(xs>0 && ys>0 && zs>0);
    UINT32 z1 = z0+zs, y1 = y0+ys, x1 = x0+xs;
    
    double frac_x = 1.0/double(xs), frac_y = 1.0/double(ys), frac_z = 1.0/double(zs);
    double sum = 0;
    for(UINT32 z=z0; z<z1; z++){
        double sum1 = 0;
        for(UINT32 y=y0; y<y1; y++){
            double sum2 = 0;
            for(UINT32 x=x0; x<x1; x++){
                sum2 += this->GetData(y, x, z);
            }
            sum1 += sum2*frac_x;
        }
        sum += sum1*frac_y;
    }
    return BT(sum*frac_z);
}


#endif
