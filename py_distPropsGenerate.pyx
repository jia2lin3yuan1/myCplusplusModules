cimport cython
cimport numpy as np

from libcpp.vector cimport vector
#from libc.stdlib cimport malloc

import numpy as np

np.import_array()

cdef extern from "ProposalGeneratePy.hxx":
    cdef vector[float] ProposalGenerate(unsigned int *imgInfo, float *distVec, float *semVec);
            

@cython.boundscheck(False)
@cython.wraparound(False)
def py_DistPropsGenerate(np.ndarray[unsigned int, ndim=1, mode='c'] imgInfo not None, 
                         np.ndarray[float, ndim=1, mode='c'] distArr not None, 
                         np.ndarray[float, ndim=1, mode='c'] semArr not None): 
    print '**calling C function in cython.'
    cdef vector[float] label_arr;
    label_arr = ProposalGenerate(&imgInfo[0], &distArr[0], &semArr[0])

    return label_arr

