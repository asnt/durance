import numpy as np
import pytest

import durance.measures
import durance.noise


@pytest.mark.xfail(
    strict=False,
    reason="DFA computation is approximate and variable due to short input"
    "signal (to lower test runtime) and random noise.")
def test_dfa_batch_gaussian_noise():
    trials = 5
    length = 2**9
    shape = (trials, length)
    signals = durance.noise.gaussian(shape=shape)

    alphas = [
        durance.measures.dfa_batch(signal, scale_min=2**4, scale_max=2**7)
        for signal in signals
    ]
    alpha = np.median(alphas)

    expected_alpha = 0.50
    loose_relative_tolerance = 1e-1
    assert np.allclose(alpha, expected_alpha, rtol=loose_relative_tolerance)


def test_dfa_batch_correlated():
    length = 2**9
    signal = np.arange(length)

    alphas = durance.measures.dfa_batch(
        signal,
        scale_min=2**4,
        scale_max=2**7,
    )
    alpha = np.mean(alphas)

    expected_alpha_lower_bound = 0.50
    assert expected_alpha_lower_bound < alpha


def test_dfa_batch_brownian_noise():
    trials = 5
    length = 2**9
    signals = durance.noise.brownian(shape=(trials, length))

    alphas = [
        durance.measures.dfa_batch(signal, scale_min=2**4, scale_max=2**7)
        for signal in signals
    ]
    alpha = np.mean(alphas)

    expected_alpha = 1.50
    loose_relative_tolerance = 1e-1
    assert np.allclose(alpha, expected_alpha, rtol=loose_relative_tolerance)


def _make_fractional_brownian_motion(size: int, beta: float) -> np.ndarray:
    """Generate approximate factional Browian motion.

    Noise signal with 1/f^beta power spectrum.
    Approximated with an inverse FFT on a random spectrum proportional to
    1/f^(beta/2) at frequency f [1].

    References
    ----------
    .. [1] Saupe, Dietmar. "Algorithms for random fractals." The science of
           fractal images. Springer, New York, NY, 1988. 71-136.
    """

    def _is_power_of_2(n: float) -> bool:
        return np.log2(n).is_integer()

    assert _is_power_of_2(size)
    assert 1 <= beta <= 3

    rng = np.random.default_rng()
    size2 = size // 2
    magnitude = (1 + np.arange(size2))**(-beta / 2) * rng.normal(size=size2)
    phase = 2 * np.pi * rng.uniform(size=size2)
    freq = np.zeros((size, ), dtype=complex)
    freq[:size2] = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    freq[size2 + 1:] = np.conj(freq[1:size2][::-1])
    freq[size2] = freq[size2].real
    noise = np.fft.ifft(freq)
    return noise.real


def _fbm_alpha_to_beta(alpha: float) -> float:
    """Convert alpha exopnent of DFA to beta exponent of fBm.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis
    """
    return 2 * alpha - 1


@pytest.mark.xfail(
    strict=False,
    reason="DFA computation is approximate and variable due to short input"
    "signal (to lower test runtime) and random noise.")
@pytest.mark.parametrize("beta", [1, 2, 3])
def test_dfa_on_fractional_brownian_motion(beta):
    trials = 10
    size = 2**10
    signals = [
        _make_fractional_brownian_motion(size, beta) for _ in range(trials)
    ]

    alphas = [durance.measures.dfa(signal) for signal in signals]
    alpha = np.mean(alphas)
    computed_beta = _fbm_alpha_to_beta(alpha)

    loose_relative_tolerance = 2e-1
    assert np.allclose(computed_beta, beta, rtol=loose_relative_tolerance)


@pytest.mark.xfail(
    strict=False,
    reason="DFA computation is approximate and variable due to short input"
    "signal (to lower test runtime) and random noise.")
@pytest.mark.parametrize("beta", [1, 2, 3])
def test_dfa_batch_on_fractional_brownian_motion(beta):
    trials = 5
    size = 2**9
    signals = [
        _make_fractional_brownian_motion(size, beta) for _ in range(trials)
    ]

    alphas = [durance.measures.dfa_batch(signal) for signal in signals]
    alpha = np.mean(alphas)
    computed_beta = _fbm_alpha_to_beta(alpha)

    loose_relative_tolerance = 2e-1
    assert np.allclose(computed_beta, beta, rtol=loose_relative_tolerance)
