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

def dimnensionaless_2d(n1, n2, x1, x2):
    return dimensionless_basis_function(n1, x1) * dimensionless_basis_function(n2, x2)

def dimensional_2d(I_n, n, x, γ):
    return I_n[0] * dimensional_basis_function(n[0], x[0], γ) * I_n[1] * dimensional_basis_function(n[1], x[1], γ)



# optimizing functions

# https://arxiv.org/pdf/astro-ph/0608369.pdf, eq 8
