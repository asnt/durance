from typing import Optional

import numpy as np


def gaussian(shape: tuple,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()
    gaussian_ = rng.normal(size=shape)
    return gaussian_


def brownian(shape: tuple,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate Brownian noise by integration of Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()
    gaussian_ = gaussian(shape=shape, rng=rng)
    brownian_ = np.cumsum(gaussian_, axis=-1) / shape[-1]
    return brownian_
