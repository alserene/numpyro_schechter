import jax
import jax.numpy as jnp
from jax import lax, jit
from jax.scipy.special import gamma, gammaincc
from jax._src.numpy.util import promote_args_inexact
from jax._src.typing import Array, ArrayLike

jax.config.update("jax_enable_x64", True)


def schechter_mag(phi_star, M_star, alpha, M):
    """
    Schechter density in magnitude space (unnormalised).
    """
    # M: absolute magnitude
    x = 10 ** (0.4 * (M_star - M))
    return (
        0.4 * jnp.log(10.0)
        * phi_star
        * x ** (alpha + 1)
        * jnp.exp(-x)
    )


def s_positive(s: ArrayLike, x: ArrayLike) -> Array:
    """
    Regularized upper incomplete gamma function * Gamma(s)
    """
    return gamma(s) * gammaincc(s, x)


def compute_gamma(s: ArrayLike, x: ArrayLike) -> Array:
    def recur(gamma_val, s, x):
        return (gamma_val - x ** s * jnp.exp(-x)) / s

    def compute_recurrence(carry, _):
        gamma_val, s = carry
        new_gamma = lax.cond(
            jnp.isinf(gamma_val),
            lambda _: jnp.inf,
            lambda _: recur(gamma_val, s - 1, x),
            operand=None
        )
        return (new_gamma, s - 1), new_gamma

    recur_depth = 3
    s_start = s + 3
    gamma_start = s_positive(s_start, x)

    initial_carry = (gamma_start, s_start)
    result, _ = lax.scan(compute_recurrence, initial_carry, None, length=recur_depth)

    return result[0]


@jit
def custom_gammaincc(s: ArrayLike, x: ArrayLike) -> Array:
    """
    Computes Γ(s, x) using a recurrence-based method.
    Valid for real s (non-integers in -3 < s < 3) and x > 0.
    """
    s, x = promote_args_inexact("custom_gammaincc", s, x)
    return compute_gamma(s, x)
