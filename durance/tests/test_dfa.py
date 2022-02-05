import numpy as np
import pytest

import durance.measures


def test_dfa_gaussian_noise():
    trials = 10
    length = 2 ** 10
    rng = np.random.default_rng()
    signals = rng.normal(size=(trials, length))
    alphas = [durance.measures.dfa(signal) for signal in signals]
    alpha = np.mean(alphas)
    expected_alpha = 0.50
    # Ensure enough tolerance because estimation of alpha is approximate.
    assert np.allclose(expected_alpha, alpha, rtol=1e-1)


def test_dfa_correlated():
    length = 2 ** 10
    signal = np.arange(length)
    alpha = durance.measures.dfa(signal)
    expected_alpha_lower_bound = 0.50
    assert expected_alpha_lower_bound < alpha


def _make_brownian_noise(shape, rng=None):
    """Generate Brownian noise by integration of Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()
    gaussian = rng.normal(size=shape)
    brownian = np.cumsum(gaussian, axis=-1) / shape[-1]
    return brownian


def test_dfa_brownian_noise():
    trials = 10
    length = 2 ** 10
    signals = _make_brownian_noise(shape=(trials, length))
    alphas = [durance.measures.dfa(signal) for signal in signals]
    alpha = np.mean(alphas)
    expected_alpha = 1.50
    # Ensure enough tolerance because estimation of alpha is approximate.
    assert np.allclose(expected_alpha, alpha, rtol=1e-1)
