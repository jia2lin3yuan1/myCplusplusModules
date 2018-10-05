cimport cython
cimport numpy as np

from libcpp.vector cimport vector
#from libc.stdlib cimport malloc

import numpy as np

np.import_array()

cdef extern from "MergeMain_py.hxx":
    cdef cppclass OutData:
        OutData() except +

        vector[double] labels
            
    cdef void ProposalGenerate(double *imgInfo, double *predVec, int *oversegVec, OutData *mydata);
            

@cython.boundscheck(False)
@cython.wraparound(False)
def py_merge(np.ndarray[double, ndim=1, mode='c'] imgInfo not None, 
                         np.ndarray[double, ndim=1, mode='c'] predArr not None, 
                         np.ndarray[int, ndim=1, mode='c'] oversegArr not None): 
    print '**calling C function in cython.'
    cdef OutData *mydata = new OutData()
    ProposalGenerate(&imgInfo[0], &predArr[0],  &oversegArr[0], mydata)
    
    print '** finished of C.'

    return mydata.labels

