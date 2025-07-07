import pytest
import jax
import jax.numpy as jnp
from sympy import uppergamma

from numpyro_schechter.math_utils import custom_gammaincc, SUPPORTED_ALPHA_DOMAIN_DEPTHS


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 5.0, 10.0, 30.0])
@pytest.mark.parametrize("s", [-2.95, -1.2, 0.5, 1.7, 2.99])
def test_custom_gammaincc_against_sympy(s, x):
    jax_result = float(custom_gammaincc(s, x))
    sympy_result = uppergamma(s, x).evalf()

    assert jax_result == pytest.approx(float(sympy_result), rel=1e-3)


@pytest.mark.parametrize("s", [-0.001, 0.999])
def test_non_integer_s_near_integer(s):
    val = custom_gammaincc(s, 1.0)
    assert jnp.isfinite(val)


@pytest.mark.parametrize("depth", SUPPORTED_ALPHA_DOMAIN_DEPTHS)
def test_custom_gammaincc_supported_depths(depth):
    val = custom_gammaincc(1.5, 2.0, recur_depth=depth)
    assert jnp.isfinite(val)


def test_invalid_depth_raises():
    with pytest.raises(ValueError, match="Unsupported recur_depth"):
        custom_gammaincc(1.5, 1.0, recur_depth=12)


def test_grad():
    grad_fn = jax.grad(lambda s: custom_gammaincc(s, 2.0))
    grad_val = grad_fn(-1.2)
    assert jnp.isfinite(grad_val)
