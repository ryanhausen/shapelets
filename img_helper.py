#https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/

import numpy as np
from astropy.io import fits

def raw_moment(img, x, y, i, j):
    return (img[y,x] * x**i * y**j).sum()
    
def img_center(img, src_map):
    y, x = np.where(src_map)
    moment = lambda i, j: raw_moment(img, x, y, i, j)

    m00 = moment(0, 0)
    m01 = moment(0, 1)
    m10 = moment(1, 0)

    x_centroid = m10 / m00
    y_centroid = m01 / m00

    return (y_centroid, x_centroid)
    
def moments_cov(img, src_map):
    y, x = np.where(src_map)
    moment = lambda i, j: raw_moment(img, x, y, i, j)

    m00 = moment(0, 0)
    m10 = moment(1, 0)
    m01 = moment(0, 1)
    x_centroid = m10 / m00
    y_centroid = m01 / m00
    u11 = (moment(1, 1) - x_centroid * m01) / m00
    u20 = (moment(2, 0) - x_centroid * m10) / m00
    u02 = (moment(0, 2) - y_centroid * m01) / m00
    cov = np.array([[u20, u11], [u11, u02]])
    return cov
    
img = fits.getdata('GDS_deep2_4135_h.fits')
segmap = fits.getdata('GDS_deep2_4135_segmap.fits')
src_map = segmap==4135

cy, cx = img_center(img, src_map)

cov = moments_cov(img, src_map)
evals, evecs = np.linalg.eig(cov)
pairs = {}
for i in range(2):
    pairs[evals[i]] = evecs[:, i]

eval_maj = evals.max()
eval_min = evals.min()

major_x, major_y = pairs[eval_maj]  # Eigenvector with largest eigenvalue
minor_x, minor_y = pairs[eval_min]

scale = np.sqrt(eval_maj) * np.array([-1.0, 1.0])
major_x_line = scale * major_x + x
major_y_line = scale * major_y + y

scale = np.sqrt(eval_min) * np.array([-1.0, 1.0])
minor_x_line = scale * minor_x + x
minor_y_line = scale * minor_y + y

theta = 0.5 * np.arctan((2 * cov[0,1])/(cov[0,0] - cov[1,1]))
theta = np.rad2deg(theta)
print(f'Theta:{theta}')




