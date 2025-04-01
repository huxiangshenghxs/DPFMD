import numpy as np
from scipy.optimize import approx_fprime

def generate_dhtm(nl:int) -> np.ndarray:
    l = np.arange(1, nl + 1).reshape(-1, 1)
    l1 = np.arange(0, nl + 1)
    denominator = l1 - l + 0.5
    dhtm = 1.0 / denominator
    return dhtm

def grad_Gamma2d(u:np.ndarray, Gamma2d, gamma_c):
    assert u.shape[1] == 2, "u should be a 2D array with shape (nl, 2)"
    nl = u.shape[0]
    eps = np.sqrt(np.finfo(float).eps)
    f = np.vstack([approx_fprime(u[l], lambda x:Gamma2d(x, gamma_c), [eps, eps]) for l in range(nl)])
    return f
