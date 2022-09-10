import numpy as np
import pytest

import durance.measures
import durance.noise


def test_dfa_gaussian_noise():
    trials = 10
    length = 2**10
    shape = (trials, length)
    signals = durance.noise.gaussian(shape=shape)

    alphas = [durance.measures.dfa(signal) for signal in signals]
    alpha = np.mean(alphas)

    expected_alpha = 0.50
    loose_tolerance = 1e-1
    assert np.allclose(expected_alpha, alpha, rtol=loose_tolerance)


def test_dfa_correlated():
    length = 2**10
    signal = np.arange(length)

    alpha = durance.measures.dfa(signal)

    expected_alpha_lower_bound = 0.50
    assert expected_alpha_lower_bound < alpha


def test_dfa_brownian_noise():
    trials = 10
    length = 2**10
    shape = (trials, length)
    signals = durance.noise.brownian(shape=shape)

    alphas = [durance.measures.dfa(signal) for signal in signals]
    alpha = np.mean(alphas)

    expected_alpha = 1.50
    loose_tolerance = 1e-1
    assert np.allclose(expected_alpha, alpha, rtol=loose_tolerance)
