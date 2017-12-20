import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import fmin

import shapelet_helper as sh
import img_helper as ih

img = fits.getdata('GDS_deep2_4135_h.fits')
segmap = fits.getdata('GDS_deep2_4135_segmap.fits')

#plt.figure()
#plt.imshow(img)
#plt.figure()
#plt.imshow(ih.rotate_img(img, segmap==4135))
#plt.show()

xc = [43, 43]
V = np.var(img[segmap==0].flatten())
V = np.diag(np.ones([84*84])*V)
V = np.linalg.inv(V)

img = ih.rotate_img(img, segmap==4135)
#img_f = np.array([img_f, img_f])


def best_gamma(gamma):
    gamma = gamma[0]
    opts, (n1_map, n2_map) = sh.solve_shapelet_coefficients(img, xc, gamma, V)
    I_n = [(opts[n1_map[n1]], opts[n2_map[n2]]) for n1, n2 in sh.TOP_SHAPELETS]

    fit = sh.goodness_of_fit(img, sh.TOP_SHAPELETS, I_n, xc, gamma, V).sum()
    print(fit, gamma)

    #return (fit-1)**2
    return fit


gamma = fmin(best_gamma, [1.0])

rotated_img = ih.rotate_img(img, segmap==4135)
opts, (n1_map, n2_map) = sh.solve_shapelet_coefficients(rotated_img,
                                                        xc,
                                                        gamma,
                                                        V)

I_n = [(opts[n1_map[n1]], opts[n2_map[n2]]) for n1, n2 in sh.TOP_SHAPELETS]


sh_img = sh.shapelet_reconstruction(img.shape, xc, sh.TOP_SHAPELETS, I_n, gamma)

plt.figure()
plt.imshow(img, cmap='gray')

plt.figure()
plt.imshow(sh_img, cmap='gray')

plt.figure()
plt.imshow(img-sh_img, cmap='gray')

plt.show()










#opts = sh.solve_shapelet_coefficients(img, xc, gamma)
#I_n = [(o,o) for o in opts]



#I_ns = [(opts[i], opts[i+1]) for i in list(range(20))[::2]]
#gamma = opts[-1]

#i_recov = sh.shapelet_reconstruction(img.shape, xc, sh.TOP_SHAPELETS, I_ns, gamma)

#plt.figure()
#plt.imshow(i_recov, cmap='gray')

#plt.show()
