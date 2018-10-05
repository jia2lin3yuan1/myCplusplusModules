import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mImage
import argparse

import sys
sys.path.insert(0, '/home/yuanjial/Projects/Python-pure/datalayer/utils/')
from labelcolor_voc import ColorMap

parser = argparse.ArgumentParser(description='specify output file name.')
parser.add_argument('--o', dest='outPath', type=str, default='notsave', help='define output path for saving')
args = parser.parse_args()


# Python saving by:: np.savetxt('fname.csv', rgbI[...,0], delimiter=',', fmt='%1.5e')

'''
cmap     = ColorMap(label_num=512)
ta = np.loadtxt('output/test.csv', delimiter=',')


print "::Showing image has shape and vale::"
print ta.shape, np.unique(ta)

ta_clr = cmap.convert_label2rgb(ta.astype(np.uint32))

if('notsave' in args.outPath):
    plt.imshow(ta)
    plt.show()
else:
    mImage.imsave(args.outPath, ta)
'''

ta = np.loadtxt('output/before_merge.csv', delimiter=',')
tb = np.loadtxt('output/after_merge.csv', delimiter=',')

fig, ax = plt.subplots(1,2)
ax[0].imshow(ta)
ax[1].imshow(tb)
plt.show()

