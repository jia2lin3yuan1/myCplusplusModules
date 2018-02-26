#ifndef _MERGER_SUPER_PIXEL_HXX
#define _MERGER_SUPER_PIXEL_HXX

#include "utils/graph.hxx"

class Dist_SuperPixel:public Supix{
protected:
    vector<UINT32> m_borderH;
    vector<UINT32> m_borderV;

public:
    Dist_SuperPixel():Supix(){}
};

template<typename NODE=Dist_SuperPixel, typename EDGE=Edge, typename BORDER=BndryPix>
class SuperPixelMerger:public graph<NODE, EDGE, BORDER>{

};



#endif
