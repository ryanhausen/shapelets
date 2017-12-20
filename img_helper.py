#https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
#https://en.wikipedia.org/wiki/Image_moment
import math
import numpy as np
from astropy.io import fits
from PIL import Image


def raw_moment(img, x, y, i, j):
    return (img[y,x] * x**i * y**j).sum()

def central_moment(img, x, xc, y, yc, p, q):
    return (img[y,x] * (x-xc)**p * (y-yc)**q).sum()

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

def get_theta(img, src_map):
    cy, cx = img_center(img, src_map)
    cov = moments_cov(img, src_map)
    evals, evecs = np.linalg.eig(cov)

    pairs = {}
    for i in range(2):
        pairs[evals[i]] = evecs[:, i]

    major_x, major_y = pairs[evals.max()]
    theta = np.arctan(major_y/major_x)
    return theta

def get_params(img, src_map):
    y, x = np.where(src_map)

    # raw moments
    M = lambda i, j: raw_moment(img, x, y, i, j)
    M00 = M(0, 0)
    M10 = M(1, 0)
    M01 = M(0, 1)

    xc = M10/M00
    yc = M01/M00

    # cenral moments
    μ = lambda p, q: central_moment(img, x, xc, y, yc, p, q)
    μ11 = μ(1, 1)
    μ20 = μ(2, 0)
    μ02 = μ(0, 2)

    # second order central moments
    μμ = lambda p, q: μ(p, q) / μ(0, 0)

    μμ20 = μμ(2, 0)
    μμ02 = μμ(0, 2)
    μμ11 = μμ(1, 1)

    # angle
    Θ = 0.5 * math.arctan((2 * μμ11) / (μμ20 - μμ02))

    # axis ratio
    cov = np.array([
        [μμ20, μμ11],
        [μμ11, μμ02]
    ])

    evals, _ = np.linalg.eig(cov)
    axis_ratio = np.sqrt(evals.min()/evals.max())





    return Θ, axis_ratio


def _translate(u,v):
    return np.array([
            [1.0, 0.0, float(u)],
            [0.0, 1.0, float(v)],
            [0.0, 0.0, 1.0]
        ])

def _rotate(angle):
    angle = -math.radians(angle)
    cos = math.cos(angle)
    sin = math.sin(angle)

    return np.array([
            [cos, sin, 0.0],
            [-sin, cos, 0.0],
            [0.0, 0.0, 1.0]
        ])


def rotate_img(img, src_map):
    cy, cx = img_center(img, src_map)
    to_origin = _translate(cy, cx)
    rotate = _rotate(-np.rad2deg(get_theta(img, src_map)))
    recenter = _translate(-cy, -cx)

    trans = to_origin.dot(rotate).dot(recenter)
    trans = tuple(trans.flatten()[:6])
    tmp = Image.fromarray(img)
    tmp = tmp.transform((84,84), Image.AFFINE, data=trans, resample=Image.BILINEAR)
    img = np.asarray(tmp)
    return img
