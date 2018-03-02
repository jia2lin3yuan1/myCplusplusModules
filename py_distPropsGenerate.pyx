cimport cython
cimport numpy as np

from libcpp.vector cimport vector
from libc.stdlib cimport malloc

import numpy as np

np.import_array()

cdef extern from "ProposalGeneratePy.hxx":
    cdef void ProposalGenerate(vector[unsigned int] &imgInfo, const vector[float] &distVec, const vector[float] &semVec, unsigned int *labelArr);
            

@cython.boundscheck(False)
@cython.wraparound(False)
def py_DistPropsGenerate(imgInfo, distArr, semArr, np.ndarray[unsigned int, ndim=1, mode='c'] labelArr not None):
    print '**calling C function in cython.'
    ProposalGenerate(<vector[unsigned int]&> imgInfo, <const vector[float]&> distArr, <const vector[float]&> semArr, &labelArr[0])

    return labelArr

