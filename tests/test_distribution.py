import jax.numpy as jnp
import numpyro
import pytest
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random

from numpyro_schechter.distribution import SchechterMag, DoubleSchechterMag
from numpyro_schechter.math_utils import SUPPORTED_ALPHA_DOMAIN_DEPTHS


# Tests for Single Schechter ------------------------
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

def test_schechter_log_prob_with_poisson():
    mag_obs = jnp.linspace(-23, -20, 5)
    d = SchechterMag(alpha=-1.2, M_star=-21.0, logphi=-2.5, mag_obs=mag_obs,
                     include_poisson_term=True, volume=6e5)
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


# Tests for Double Schechter ------------------------
def test_doubleschechtermag_inference_runs():
    rng_key = random.PRNGKey(0)
    mag_data = jnp.linspace(-22.5, -20.5, 50)

    def model(mag_obs):
        alpha1 = numpyro.sample("alpha1", dist.Uniform(-3.0, 1.0))
        M_star1 = numpyro.sample("M_star1", dist.Uniform(-24.0, -20.0))
        logphi1 = numpyro.sample("logphi1", dist.Normal(-3.0, 1.0))
        alpha2 = numpyro.sample("alpha2", dist.Uniform(-3.0, 1.0))
        M_star2 = numpyro.sample("M_star2", dist.Uniform(-24.0, -20.0))
        logphi2 = numpyro.sample("logphi2", dist.Normal(-3.0, 1.0))
        numpyro.sample("mag", DoubleSchechterMag(alpha1, M_star1, logphi1,
                                                 alpha2, M_star2, logphi2,
                                                 mag_obs), obs=mag_obs)

    mcmc = MCMC(NUTS(model), num_warmup=10, num_samples=20)
    mcmc.run(rng_key, mag_obs=mag_data)
    samples = mcmc.get_samples()
    assert all(k in samples for k in ["alpha1", "M_star1", "logphi1", "alpha2", "M_star2", "logphi2"])

def test_doubleschechtermag_log_prob_finite():
    mag_obs = jnp.linspace(-23, -20, 5)
    d = DoubleSchechterMag(alpha1=-1.2, M_star1=-21.0, logphi1=-2.5,
                           alpha2=-0.5, M_star2=-20.5, logphi2=-2.8,
                           mag_obs=mag_obs)
    logp = d.log_prob(mag_obs)
    assert jnp.all(jnp.isfinite(logp))


def test_doubleschechtermag_log_prob_with_poisson():
    mag_obs = jnp.linspace(-23, -20, 5)
    d = DoubleSchechterMag(alpha1=-1.2, M_star1=-21.0, logphi1=-2.5,
                           alpha2=-0.5, M_star2=-20.5, logphi2=-2.8,
                           mag_obs=mag_obs, include_poisson_term=True, volume=6e5)
    logp = d.log_prob(mag_obs)
    assert jnp.all(jnp.isfinite(logp))
