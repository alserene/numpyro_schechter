import jax.numpy as jnp
import numpyro
import pytest
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random

from numpyro_schechter.distribution import SchechterMag
from numpyro_schechter.math_utils import SUPPORTED_ALPHA_DOMAIN_DEPTHS


def test_schechtermag_inference_runs():
    rng_key = random.PRNGKey(0)
    mag_data = jnp.linspace(-22.5, -20.5, 50)  # synthetic data

    def model(mag_obs):
        alpha = numpyro.sample("alpha", dist.Uniform(-3.0, 1.0))
        M_star = numpyro.sample("M_star", dist.Uniform(-24.0, -20.0))
        logphi = numpyro.sample("logphi", dist.Normal(-3.0, 1.0))
        numpyro.sample("mag", SchechterMag(alpha, M_star, logphi, mag_obs), obs=mag_obs)

    mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=20)
    mcmc.run(rng_key, mag_obs=mag_data)
    samples = mcmc.get_samples()
    assert "alpha" in samples and "M_star" in samples and "logphi" in samples


def test_schechter_log_prob_finite():
    mag_obs = jnp.linspace(-23, -20, 5)
    d = SchechterMag(alpha=-1.2, M_star=-21.0, logphi=-2.5, mag_obs=mag_obs)
    logp = d.log_prob(mag_obs)
    assert jnp.all(jnp.isfinite(logp))


@pytest.mark.parametrize("depth", SUPPORTED_ALPHA_DOMAIN_DEPTHS)
def test_log_prob_finite_for_supported_depth(depth):
    mag_data = jnp.linspace(-22.5, -20.5, 10)
    dist = SchechterMag(alpha=-1.0, M_star=-21.0, logphi=-3.0, mag_obs=mag_data, alpha_domain_depth=depth)
    logp = dist.log_prob(mag_data)
    assert jnp.all(jnp.isfinite(logp)), f"logp failed for depth {depth}"


def test_invalid_alpha_domain_depth():
    mag_data = jnp.linspace(-22.5, -20.5, 10)
    with pytest.raises(ValueError, match="Unsupported recur_depth"):
        SchechterMag(alpha=-1.0, M_star=-21.0, logphi=-3.0, mag_obs=mag_data, alpha_domain_depth=12)


def test_supported_depths_list():
    assert SchechterMag.supported_depths() == SUPPORTED_ALPHA_DOMAIN_DEPTHS
