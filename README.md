# numpyro_schechter

**Schechter galaxy luminosity distribution for NumPyro**

<p align="center">
  <img src="https://raw.githubusercontent.com/alserene/numpyro_schechter/main/docs/assets/logo.png" alt="numpyro_schechter logo" width="300"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/">
    <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
  </a>
  <a href="https://numpyro-schechter.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/numpyro-schechter/badge/?version=latest" alt="Docs Status">
  </a>
  <a href="https://pypi.org/project/numpyro-schechter/">
    <img src="https://img.shields.io/pypi/v/numpyro-schechter.svg" alt="PyPI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  </a>
  <!-- <a href="https://github.com/alserene/numpyro_schechter/actions/workflows/tests.yml">
    <img src="https://github.com/alserene/numpyro_schechter/actions/workflows/tests.yml/badge.svg" alt="Tests">
  </a> -->
</p>

---

## Overview

`numpyro_schechter` provides a NumPyro-compatible probability distribution for Bayesian inference with Schechter luminosity functions in absolute magnitude space. Built for astronomers and statisticians, it includes a JAX-compatible custom implementation of the upper incomplete gamma function, enabling stable and differentiable modelling within probabilistic programming frameworks.

**Note:** Due to the custom implementation of the incomplete gamma function, the distribution is **only valid when `alpha + 1` is in the range (-3, 3) and is non-integer**. Users are responsible for ensuring parameters fall within this valid range.

---

## Installation

You can install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/alserene/numpyro_schechter.git
```

Or from PyPI, install via:

```bash
pip install numpyro_schechter
```

---

## Usage

Here is a minimal example showing how to instantiate and use the `SchechterMag` distribution:

```python
import jax
import jax.numpy as jnp
from numpyro_schechter.distribution import SchechterMag

# Example parameters (ensure alpha + 1 ∈ (-3, 3) and non-integer)
alpha = -1.3
M_star = -20.0
logphi = jnp.log(1e-3)

# Observed magnitude range
mag_obs = jnp.linspace(-24, -16, 100)

# Instantiate the distribution
schechter_dist = SchechterMag(alpha=alpha, M_star=M_star, logphi=logphi, mag_obs=mag_obs)

# Evaluate log probability at some magnitude value
mag_val = -21.0
logp = schechter_dist.log_prob(mag_val)
print(f"log probability at M={mag_val}: {logp}")

# Note: Sampling is not implemented
```

For detailed usage and API documentation, please visit the [Documentation](https://numpyro-schechter.readthedocs.io/).

---

## Development

If you want to contribute or develop locally:

```bash
git clone https://github.com/alserene/numpyro_schechter.git
cd numpyro_schechter
poetry install
poetry run pytest
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

Created by Alice — [aserene@swin.edu.au](mailto:aserene@swin.edu.au)

---

*Happy modelling!*
