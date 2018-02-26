import numpy as np
from matplotlib import pyplot as plt

# Python saving by:: np.savetxt('fname.csv', rgbI[...,0], delimiter=',', fmt='%1.5e')

ta = np.loadtxt('output/test_graph.csv', delimiter=',')
print ta.shape, np.unique(ta)

plt.imshow(ta)
plt.show()
'''

ta = np.loadtxt('output/test_extend.csv', delimiter=',')
tb = np.loadtxt('output/test_shrink.csv', delimiter=',')

fig, ax = plt.subplots(1,2)
ax[0].imshow(ta)
ax[1].imshow(tb)
plt.show()
'''
