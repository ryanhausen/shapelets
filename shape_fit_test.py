import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import fmin

import shapelet as sh
import img_helper as ih

img = fits.getdata('GDS_deep2_4135_h.fits')
segmap = fits.getdata('GDS_deep2_4135_segmap.fits')

# plt.figure()
# plt.imshow(img)

coeffs, γ = sh.fit_shapelets(img, segmap==4135, ns=sh.TOP_SHAPELETS)
print(γ)
print(coeffs)

sh_img = sh.construct_img((84,84), coeffs, γ, ns=sh.TOP_SHAPELETS)

#plt.figure()
#plt.imshow(ih.rotate_img(img, segmap==4135))
#plt.show()

#sh_img = sh.shapelet_reconstruction(img.shape, xc, sh.TOP_SHAPELETS, I_n, gamma)

plt.figure()
plt.imshow(img, cmap='gray')

plt.figure()
plt.imshow(sh_img, cmap='gray')

plt.figure()
plt.imshow(img-sh_img, cmap='gray')

plt.show()