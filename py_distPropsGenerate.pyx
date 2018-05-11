cimport cython
cimport numpy as np

from libcpp.vector cimport vector
#from libc.stdlib cimport malloc

import numpy as np

np.import_array()

cdef extern from "ProposalGeneratePy.hxx":
    cdef cppclass OutData:
        OutData() except +

        vector[double] labels
        vector[double] merge_flag
            
    cdef void ProposalGenerate(unsigned int *imgInfo, double *distVec, double *semVec, unsigned int *instVec, OutData *mydata);
            

@cython.boundscheck(False)
@cython.wraparound(False)
def py_DistPropsGenerate(np.ndarray[unsigned int, ndim=1, mode='c'] imgInfo not None, 
                         np.ndarray[double, ndim=1, mode='c'] distArr not None, 
                         np.ndarray[double, ndim=1, mode='c'] semArr not None, 
                         np.ndarray[unsigned int, ndim=1, mode='c'] instArr not None): 
    print '**calling C function in cython.'
    cdef OutData *mydata = new OutData()
    ProposalGenerate(&imgInfo[0], &distArr[0], &semArr[0], &instArr[0], mydata)
    
    print '** finished of C.'

    return mydata.labels, mydata.merge_flag

