import numpy as np
from matplotlib import pyplot as plt

ta = np.loadtxt('test.csv', delimiter=',')
print ta.shape, np.unique(ta)

plt.imshow(ta)
plt.show()
'''

ta = np.loadtxt('test_extend.csv', delimiter=',')
tb = np.loadtxt('test_shrink.csv', delimiter=',')

fig, ax = plt.subplots(1,2)
ax[0].imshow(ta)
ax[1].imshow(tb)
plt.show()
'''
