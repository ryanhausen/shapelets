from itertools import combinations
import numpy as np
from scipy.misc import factorial
import img_helper as ih

# https://arxiv.org/pdf/astro-ph/0307395.pdf, table 1
TOP_SHAPELETS = [(0,0), (4,0), (2,0), (0,4), (6,0),
                 (2,2), (0,2), (8,0), (0,1), (1,0)]

ALL_SHAPELETS = np.array(list(combinations(range(11), 2)))

def _hermite(n, x):
    if n==0:
        return 1
    elif n==1:
        return 2*x
    elif n==2:
        return 4*x**2 - 2
    elif n==3:
        return 8*x**3 - 12*x
    elif n==4:
        return 16*x**4 + 48*x**2 + 12
    elif n==5:
        return 32*x**5 - 160*x**3 + 120*x
    elif n==6:
        return 64*x**6 - 480*x**4 + 720*x**2 - 120
    elif n==7:
        return 128*x**7 - 1344*x**5 + 3360*x**3 - 1680*x
    elif n==8:
        return 256*x**8 - 3584*x**6 + 13440*x**4 - 13440*x**2 + 1680
    elif n==9:
        return 512*x**9 - 9216*x**7 + 48384*x**5 - 80640*x**3 + 30240
    elif n==10:
        return 1024*x**10 - 23040*x**8 +161280*x**6 - 403200*x**4 + 302400*x**2 - 30240

# https://arxiv.org/pdf/astro-ph/0307395.pdf, eq1
# $$\phi_n(x) = [2^n\pi^{\frac{1}{2}}n!]^{-\frac{1}{2}}H_n(x)e^{-\frac{x^2}{2}}$$
def dimensionless_basis_function(n, x):
    return (2**n * np.pi**0.5 * factorial(n))**-0.5 * _hermite(n,x)*np.exp(-x**2/2)

# https://arxiv.org/pdf/astro-ph/0307395.pdf, eq2
# $$B_n(x;\gamma) = \gamma^{-\frac{1}{2}}\phi_n(\gamma^{-1}x)$$
def dimensional_basis_function(n, x, gamma):
    return gamma**-0.5 * dimensionless_basis_function(n, x/gamma)

def dimensional_2d(n, x, γ):
    return (γ[0] * γ[1])**0.5 * dimensionless_basis_function(n[0], x[0]/γ[0]) * dimensionless_basis_function(n[1], x[1]/γ[1])
    #return I_n[0] * dimensional_basis_function(n[0], x[0], γ[0]) * I_n[1] * dimensional_basis_function(n[1], x[1], γ[1])

def make_M(dims, yc_xc, γ, ns=TOP_SHAPELETS):
    ms = []

    γ1, γ2 = γ
    x1, x2 = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
    x1 = (x1-yc_xc[0])
    x2 = (x2-yc_xc[1])

    for n in ns:
        basis = dimensional_2d(n, (x1, x2), γ)
        ms.append(basis.flatten())

    return np.array(ms)


from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
def _preprocess_img(img, mask):

    #img = Normalizer(norm='l1').fit_transform(img)

    return img, mask

def _postprocess_img(img, mask):
    return img, mask

def fit_shapelets(img, mask, ns=TOP_SHAPELETS):

    # do any preprocessing
    img, mask = _preprocess_img(img, mask)

    # get params from the image
    Θ, axis_ratio, γ1, γ2, yc_xc = ih.get_params(img, mask)
    print(Θ, axis_ratio, γ1, γ2, yc_xc)
    γ1, γ2 = 0.98, 0.98

    # Variance matrix V
    V = np.var(img[mask==0].flatten())
    V = np.diag(np.ones([84*84])*V)
    V = np.linalg.inv(V)

    # rotate and center image
    img = ih.rotate_img(img, mask, Θ)

    # do any post processing
    img, mask = _postprocess_img(img, mask)

    # solve coeffecients
    M = make_M((84,84), yc_xc, (γ1, γ2)).T
    I = img.flatten()[:,np.newaxis]

    MᵀVM = M.T.dot(V).dot(M)
    MᵀVM_inv = np.linalg.inv(MᵀVM)
    MᵀVM_invMᵀ = MᵀVM_inv.dot(M.T)
    MᵀVM_invMᵀV = MᵀVM_invMᵀ.dot(V)
    MᵀVM_invMᵀVI = MᵀVM_invMᵀV.dot(I)

    return MᵀVM_invMᵀVI, (γ1, γ2)

def construct_img(shape, coeffs, γ, ns=TOP_SHAPELETS):
    img = np.zeros(shape)

    ys, xs = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    ys, xs = ys-42, xs-42

    coeffs = [(c[0], c[0]) for c in coeffs]

    for I_n, n in zip(coeffs, ns):
        img += np.abs(np.prod(I_n)) * dimensional_2d(n, (ys, xs), γ)

    return img













