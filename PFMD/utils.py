import jax.numpy as jnp
from jax import grad, vmap
# from scipy.optimize import approx_fprime

def generate_dhtm(nl:int) -> jnp.ndarray:
    l = jnp.arange(1, nl + 1).reshape(-1, 1)
    l1 = jnp.arange(0, nl + 1)
    denominator = l1 - l + 0.5
    dhtm = 1.0 / denominator
    return dhtm

def grad_Gamma2d(u:jnp.ndarray, Gamma2d, gamma_c):
    assert u.shape[1] == 2, "u should be a 2D array with shape (nl, 2)"
    f = vmap(grad(Gamma2d), in_axes=(0, None))(u, gamma_c)
    return f
