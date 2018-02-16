#ifndef _DATA_TEMPLATE_HXX
#define _DATA_TEMPLATE_HXX

#include <cassert>
#include <algorithm>

#include "PlatformDefs.hxx"

template <typename BT>
class CDataTempl{
protected:
    BT *m_pBuf; // The data is arraged as: [x0y0z0, x1y0z0, ...,xNy0z0, x0y1,z0, x1y1z0, ..., xNy1z0,........., XNYMz0, 
                //                          x0y0z1, x1y0z1, ..., ....]
    UINT32 m_xDim, m_yDim, m_zDim;

    void AllocateBuf(){
        Destroy();
        m_pBuf = new BT[m_xDim*m_yDim*m_zDim];
        memset(m_pBuf, 0, m_xDim*m_yDim*m_zDim*sizeof(BT));
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
        m_pBuf = NULL;
    }
    CDataTempl(UINT32 dy, UINT32 dx=1, UINT32 dz=1){
        this->Init(dy, dx, dz);
    }
    ~CDataTempl(){
        Destroy();
        m_xDim = m_yDim = m_zDim = 0;
    }

    void Init(UINT32 dy, UINT32 dx=1, UINT32 dz=0){
        m_xDim = dx; m_yDim = dy; m_zDim = dz;
        this->AllocateBuf();
    }
    
    CDataTempl<BT> & operator=(CDataTempl<BT> &data){
        this->Init(data.GetYDim(), data.GetXDim(), data.GetZDim());
        this->Copy(data);
        return *this;
    }

    CDataTempl<BT> & operator+(CDataTempl<BT> &data){
        assert(m_xDim==data.GetXDim() && m_yDim==data.GetYDim() && m_zDim==data.GetZDim());
        CDataTempl<BT> outData(m_yDim, m_xDim, m_zDim);
        for(int z=0; z<m_zDim; z++){
            for(int y=0; y<m_yDim; y++){
                for(int x=0; x<m_xDim; x++){
                    BT val = m_pBuf[this->Coordinate2Index(y,x,z)] + data.GetData(y,x,z);
                    outData.SetData(val, y, x, z);
                }
            }
        }

        return outData;
    }


    // inline function.
    UINT32 GetXDim() const;
    UINT32 GetYDim() const;
    UINT32 GetZDim() const;
    UINT32 Coordinate2Index(UINT32 y, UINT32 x, UINT32 z) const;
    void SetData(BT val, UINT32 y, UINT32 x=0, UINT32 z=0);

    // normal member functions.
    BT GetData(UINT32 y, UINT32 x=0, UINT32 z=0){
        x = x < 0? 0 : (x>=m_xDim? m_xDim-1 : x);
        y = y < 0? 0 : (y>=m_yDim? m_yDim-1 : y);
        z = z < 0? 0 : (z>=m_zDim? m_zDim-1 : z);

        return m_pBuf[this->Coordinate2Index(y,x,z)];
    }

    void GetBulkData(BT data[], UINT32 sy, UINT32 dy, UINT32 sx=0, UINT32 dx=0, UINT32 sz=0, UINT32 dz=0){
       // todo:: 
    }

    void Copy(CDataTempl<BT> &data){
        assert(m_xDim==data.GetXDim() && m_yDim==data.GetYDim() && m_zDim==data.GetZDim());
        for(int z=0; z<m_zDim; z++){
            for(int y=0; y<m_yDim; y++){
                for(int x=0; x<m_xDim; x++){
                    m_pBuf[this->Coordinate2Index(y,x,z)] = data.GetData(y,x,z);
                }
            }
        }
    }

    void Reset(BT initVal){
        for(int z=0; z<m_zDim; z++){
            for(int y=0; y<m_yDim; y++){
                for(int x=0; x<m_xDim; x++){
                    m_pBuf[this->Coordinate2Index(y,x,z)] = initVal;
                }
            }
        }
    }




};

template <typename BT>
inline UINT32 CDataTempl<BT>::GetXDim() const {return m_xDim;}
template <typename BT>
inline UINT32 CDataTempl<BT>::GetYDim() const {return m_yDim;}
template <typename BT>
inline UINT32 CDataTempl<BT>::GetZDim() const {return m_zDim;}

template <typename BT>
inline UINT32 CDataTempl<BT>::Coordinate2Index(UINT32 y, UINT32 x, UINT32 z) const {
    return (z*m_yDim + y)*m_xDim + x;
}

template <typename BT>
inline void CDataTempl<BT>::SetData(BT val, UINT32 y, UINT32 x, UINT32 z){
    m_pBuf[this->Coordinate2Index(y,x,z)] = val;
}

#endif
