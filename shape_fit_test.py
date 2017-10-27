import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import shapelet_helper as sh

img = np.zeros([50,50])
xs, ys = np.meshgrid(np.arange(50), np.arange(50))
cxy = 25

I_n = [1.0, 1.0]
for i in range(50):
    for j in range(50):
        img[i,j] = sh.dimensional_2d(I_n, [0,0], [i-cxy, j-cxy], 1.0)

plt.imshow(img)
plt.show()
