import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.insert(0, '/home/yuanjial/Code/Python-pure/datalayer/utils/')
from labelcolor_voc import ColorMap

# Python saving by:: np.savetxt('fname.csv', rgbI[...,0], delimiter=',', fmt='%1.5e')

cmap     = ColorMap(label_num=512)
ta = np.loadtxt('output/test.csv', delimiter=',')

ta_clr = cmap.convert_label2rgb(ta.astype(np.uint32))

print "::Showing image has shape and vale::"
print ta.shape, np.unique(ta)

plt.imshow(ta_clr)
plt.show()
'''

ta = np.loadtxt('output/test_extend.csv', delimiter=',')
tb = np.loadtxt('output/test_shrink.csv', delimiter=',')

fig, ax = plt.subplots(1,2)
ax[0].imshow(ta)
ax[1].imshow(tb)
plt.show()

'''
