```markdown
# numpyro_schechter

**Schechter luminosity function distribution for NumPyro**

---

## Overview

`numpyro_schechter` provides a NumPyro-compatible probability distribution for Bayesian inference with Schechter luminosity functions in absolute magnitude space. Built for astronomers and statisticians, it includes a JAX-compatible custom implementation of the upper incomplete gamma function, enabling stable and differentiable modelling within probabilistic programming frameworks.

---

## Installation

You can install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/alserene/numpyro_schechter.git
```

Or if you publish to PyPI, install via:

```bash
pip install numpyro_schechter
```

---

## Usage

Here’s a minimal example to get you started:

```python
from numpyro_schechter import distribution

# Example: instantiate and use your distribution here
# distribution.some_function(...)
```

For detailed usage and API documentation, please visit the [Documentation](https://your-readthedocs-url.readthedocs.io).

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

*Happy modeling!*
```
