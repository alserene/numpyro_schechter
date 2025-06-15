import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random

from numpyro_schechter.distribution import SchechterMag


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
