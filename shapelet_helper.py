import math
import numpy as np
from scipy.special import factorial
from scipy.optimize import fmin

# https://en.wikipedia.org/wiki/Hermite_polynomials#Recursion_relation
# $$H_n(x)=n!\sum_{m=0}^{\lfloor\frac{n}{2}\rfloor}\frac{(-1)^m}{m!(n-2m)!}(2x)^{n-2m}$$
def hermite_polynomial(n, x):
    sum_length = n//2
    m = np.arange(sum_length+1)
    x = np.repeat(x[:,:,np.newaxis], len(m), axis=2)
    numer = np.full(m.shape, -1)**m
    denom = factorial(m) * factorial(n - 2*m)
    prod = (2*x)**(n-2*m)

    return factorial(n) * np.sum(numer/denom*prod, axis=2)

# https://arxiv.org/pdf/astro-ph/0307395.pdf, eq1
# $$\phi_n(x) = [2^n\pi^{\frac{1}{2}}n!]^{-\frac{1}{2}}H_n(x)e^{-\frac{x^2}{2}}$$
def dimensionless_basis_function(n, x):
    return (2**n * np.pi**0.5 * factorial(n))**-0.5 * hermite_polynomial(n,x)*np.exp(-x**2/2)

# https://arxiv.org/pdf/astro-ph/0307395.pdf, eq2
# $$B_n(x;\gamma) = \gamma^{-\frac{1}{2}}\phi_n(\gamma^{-1}x)$$
def dimensional_basis_function(n, x, gamma):
    return gamma**-0.5 * dimensionless_basis_function(n, gamma**-1.0 * x)

# evaluates pixel values for a single shapelet
def dimensional_2d(I_n, n, x, γ):
    return I_n[0] * dimensional_basis_function(n[0], x[0], γ) * I_n[1] * dimensional_basis_function(n[1], x[1], γ)

# generate an image using shapelets
# https://arxiv.org/pdf/astro-ph/0608369.pdf, eq.1
# $$I(\boldsymbol{x}) = I_{reco}(\boldsymbol{x}) \approx \sum_{n_1,n_2}^{n_1+n_2=n_{max}} \boldsymbol{I_nB_n}(\boldsymbol{x}-\boldsymbol{x_c};\gamma)$$
# β in the above paper is γ in the first paper
def single_shapelet_reconstruction(dims, xc, n, I_n, γ):
    """
    dims: 2d tuple-like object, size of img (dim0, dim1)
    xc  : 2d tuple-like object, center (y, x)
    n   : 2d tuple-like object, n1, n2
    I_n : 2d tuple-like object, I_n1, I_n2
    """

    ys, xs = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
    ys, xs = ys-xc[0], xs-xc[1]

    vals = dimensional_2d(I_n, n, (xs,ys), γ)

    return vals

# applies creates an image composed of shapelets and their coefficients
def shapelet_reconstruction(dims, xc, ns, I_ns, γ):
    img = np.zeros(dims)

    ys, xs = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
    ys, xs = ys-xc[0], xs-xc[1]

    for I_n, n in zip(I_ns, ns):
        img += dimensional_2d(I_n, n, (xs, ys), γ)

    return img



# optimizing functions

# https://arxiv.org/pdf/astro-ph/0608369.pdf, eq 8
# $$ \chi^2 = \frac{R(\beta,n_{max},\boldsymbol{x_c})^T\cdot V^{-1} \cdot  R(\beta,n_{max},\boldsymbol{x_c})}{n_{pixels}-n_{coeffs}}$$
# β in the above paper is γ in the first paper
def goodness_of_fit(img, ns, I_ns, xc, γ, V=None):
    I = img
    I_recov = shapelet_reconstruction(img.shape, xc, ns, I_ns, γ)
    R = I-I_recov
    n_pixels = len(img)
    n_coeffs = len(ns)

    if V:
        numer = R.T.dot(np.linalg.inv(V)).dot(R)
    else:
        numer = R.T.dot(R)
    denom = n_pixels-n_coeffs

    return numer/denom

# https://arxiv.org/pdf/astro-ph/0307395.pdf, table 1
TOP_SHAPELETS = [(0,0), (4,0), (2,0), (0,4), (6,0), (2,2), (0,2), (8,0), (0,1), (1,0)]

def make_M(dims, xc, ns, γ):
    ms = []

    x1, x2 = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
    x1 = (x1-xc[0])
    x2 = (x2-xc[1])
    I_n = (1.0, 1.0)

    #for n in ns:
    #    basis = dimensional_2d(I_n, n, (x1, x2), γ)
    #    ms.append(basis.flatten())

    x1_map, x2_map = {}, {}
    idx = 0
    for n1, n2 in ns:
        if n1 not in x1_map.keys():
            ms.append(dimensional_basis_function(n1, x1, γ).flatten())
            x1_map[n1]=idx
            idx += 1

        if n2 not in x2_map.keys():
            ms.append(dimensional_basis_function(n2, x2, γ).flatten())
            x2_map[n2]=idx
            idx += 1



    return np.array(ms).T, (x1_map, x2_map)

import matplotlib.pyplot as plt
# https://arxiv.org/pdf/astro-ph/0608369.pdf, eq 10
def solve_shapelet_coefficients(img, xc, γ, V=None):

    M, nkeys = make_M(img.shape, xc, TOP_SHAPELETS, γ)
    I = img.flatten()[:,np.newaxis]


    if V:
        V = np.inv(V)
        MᵀVM = M.T.dot(V).dot(M)
        MᵀVM_inv = np.inv(MᵀVM)
        MᵀVM_invMᵀ = MᵀVM_inv.dot(M.T)
        MᵀVM_invMᵀV = MᵀVM_invMᵀ.dot(V)
        MᵀVM_invMᵀVI = MᵀVM_invMᵀV.dot(I)

        return MᵀVM_invMᵀVI, nkeys
    else:
        MᵀM  = M.T.dot(M)
        MᵀM_inv = np.linalg.inv(MᵀM)
        MᵀM_inv_Mᵀ = MᵀM_inv.dot(M.T)
        MᵀM_inv_MᵀI = MᵀM_inv_Mᵀ.dot(I)

        return MᵀM_inv_MᵀI, nkeys
