import math
import numpy as np
from scipy.special import factorial

# https://en.wikipedia.org/wiki/Hermite_polynomials#Recursion_relation
# $$H_n(x)=n!\sum_{m=0}^{\lfloor\frac{n}{2}\rfloor}\frac{(-1)^m}{m!(n-2m)!}(2x)^{n-2m}$$
def hermite_polynomial(n, x):
    sum_length = n//2
    m = np.arange(sum_length+1)
    numer = np.full(m.shape, -1)**m
    denom = factorial(m) * factorial(n - 2*m)
    prod = (2*x)**(n-2*m)

    return factorial(n) * np.sum(numer/denom*prod)

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
        img += dimensional_2d(I_n, n, (ys, xs), γ)

    return img

# optimizing functions
# https://arxiv.org/pdf/astro-ph/0608369.pdf, eq 8
# $$ \chi^2 = \frac{R(\beta,n_{max},\boldsymbol{x_c})^T\cdot V^{-1} \cdot  R(\beta,n_{max},\boldsymbol{x_c})}{n_{pixels}-n_{coeffs}}$$
# β in the above paper is γ in the first paper
def goodness_of_fit(img, V, ns, I_ns, xc, γ):
    I = img.flatten()
    I_recov = shapelet_reconstruction(img.shape, xc, ns, I_ns, γ).flatten()
    R = I-I_recov
    n_pixels = len(img)
    n_coeffs = len(ns)

    numer = R.T.dot(np.inv(V)).dot(R)
    denom = n_pixels-n_coeffs

    return numer/denom
