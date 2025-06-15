import jax.numpy as jnp
from numpyro.distributions import Distribution, constraints
from .math_utils import custom_gammaincc, schechter_mag


class SchechterMag(Distribution):
    """
    NumPyro-compatible distribution based on the Schechter luminosity function in magnitude space.
    """
    support = constraints.real

    @property
    def has_rsample(self) -> bool:
        return False

    def __init__(self, alpha, M_star, logphi, mag_obs, validate_args=None):
        self.alpha = alpha
        self.M_star = M_star
        self.logphi = logphi
        self.phi_star = jnp.exp(logphi)
        self.mag_obs = mag_obs

        # Normalisation over observed magnitude range
        M_min, M_max = jnp.min(mag_obs), jnp.max(mag_obs)
        a = alpha + 1.0
        x_min = 10 ** (0.4 * (M_star - M_max))
        x_max = 10 ** (0.4 * (M_star - M_min))
        norm = self.phi_star * (custom_gammaincc(a, x_min) - custom_gammaincc(a, x_max))
        self.norm = jnp.where(norm > 0, norm, jnp.inf)

        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    def log_prob(self, value):
        pdf = schechter_mag(self.phi_star, self.M_star, self.alpha, value) / self.norm
        return jnp.log(pdf + 1e-30)

    def sample(self, key, sample_shape=()):
        raise NotImplementedError("Sampling not implemented for SchechterMag.")
