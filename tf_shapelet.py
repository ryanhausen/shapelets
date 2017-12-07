import numpy as np
import tensorflow as tf

def hermite_polynomial(n, x):
    p = lambda p: tf.pow(x, p)

    if n==0:
        return 1
    elif n==1:
        return 2 * x
    elif n==2:
        return 4 *  p(2) - 2
    elif n==3:
        return 8 * p(3) - 12 * x
    elif n==4:
        return 16 * p(4) - 48 * p(2) + 12
    elif n==5:
        return 32 * p(5) - 160 * p(3) + 120 * x
    elif n==6:
        return 64 * p(6) - 480 * p(4) + 720 * p(2) - 120
    elif n==7:
        return 128 * p(7) - 1344 * p(5) + 3360 * p(3) - 1680 * x
    elif n==8:
        return 256 * p(8) - 3584 * p(6) + 13440 * p(4) - 13440 * p(2) + 1680
    elif n==9:
        return 512 * p(9) - 9216 * p(7) + 48384 * p(5) - 80640 * p(3) + 30240 *x

    