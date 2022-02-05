import numpy as np


def find_inliers(
    signal: np.ndarray,
    method: str = "moving_median",
) -> np.ndarray:
    """Find samples deviating significantly from main trend of signal.

    Can filter noisy signal of heartbeat intervals (RR) into clean signal (NN).
    For example, filter out missed heartbeats.

    Parameters
    ----------
    signal : float[n]
        Signal with spurious samples.
    method : {"moving_median", "deviation", "deviation_forward"}
        Method to identify outliers.

    Returns
    -------
    bool[n]
        Mask of valid samples.
    """
    if method == "deviation":
        mask_valid = inliers_from_deviation(signal)
    elif method == "deviation_forward":
        mask_valid = inliers_from_deviation_forward(signal)
    elif method == "moving_median":
        mask_valid = inliers_from_moving_median(signal)
    elif method == "wavelet":
        raise NotImplementedError
        # XXX: Does not work. Loss of details?
        mask_valid = np.full_like(signal, True)
    else:
        raise ValueError(f"invalid method {method}")
    return mask_valid


def inliers_from_deviation_forward(signal: np.ndarray) -> np.ndarray:
    """Find valid samples in signal from forward deviation.

    Parameters
    ----------
    signal : float[n]
        Signal with spurious samples.

    Returns
    -------
    bool[n]
        Mask of valid samples.
    """
    diff_0 = np.diff(signal)
    diff_0 = np.concatenate((diff_0, [np.inf]))
    relative_variation_0 = np.abs(diff_0) / np.abs(signal)
    threshold_variation = 0.05
    mask_valid = relative_variation_0 < threshold_variation
    return mask_valid


def inliers_from_deviation(signal: np.ndarray) -> np.ndarray:
    """Find valid samples in signal from backward and forward deviations.

    Parameters
    ----------
    signal : float[n]
        Signal with spurious samples.

    Returns
    -------
    bool[n]
        Mask of valid samples.
    """
    diff_0 = np.diff(signal)
    diff_1 = np.diff(signal[::-1])[::-1]
    diff_0 = np.concatenate((diff_0, [0]))
    diff_1 = np.concatenate(([0], diff_1))
    relative_variation_0 = np.abs(diff_0) / np.abs(signal)
    relative_variation_1 = np.abs(diff_1) / np.abs(signal)
    threshold_variation = 0.05
    # threshold_variation = 0.10
    mask_valid = (
        (relative_variation_0 < threshold_variation)
        & (relative_variation_1 < threshold_variation)
    )
    return mask_valid


def inliers_from_moving_median(
    signal: np.ndarray,
    window_size: int = 31,
    method: str = "absolute",
    threshold: float = 0.015,
) -> np.ndarray:
    """Find valid samples in signal from moving median.

    Parameters
    ----------
    signal : float[n]
        Signal with suprious samples.
    window_size :
        Size of moving window.
    method : {"absolute", "quantile"}
        Method to determine inliers.
    threshold :
        Maxixum distance from moving median for an inlier.
        Maximum deviation in seconds for method 'absolute' (e.g. 0.015).
        Unitless quantile value in (0, 1) for method 'quantile' (e.g. 0.85).

    Returns
    -------
    bool[n]
        Mask of valid samples.
    """
    assert method in ("absolute", "quantile")
    if method == "quantile":
        assert 0.0 < threshold < 1.0

    pad_before = window_size // 2
    pad_after = window_size - pad_before - 1
    pad_widths = pad_before, pad_after
    signal_padded = np.pad(
        signal, pad_widths, mode="constant", constant_values=np.nan,
    )

    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    windows = sliding_window_view(signal_padded, window_size)
    # Need explicit cast to dtype float for some operations below (e.g.
    # np.nanmedian). By default, the sliding window views have dtype object.
    windows = windows.astype(float)

    medians = np.nanmedian(windows, axis=1)
    deviations = np.abs(signal - medians)

    if method == "absolute":
        threshold_ = threshold
    elif method == "quantile":
        threshold_ = np.quantile(deviations, threshold)

    mask_valid = deviations < threshold_

    return mask_valid


def inliers_from_swt(signal: np.ndarray) -> np.ndarray:
    """Find valid samples in signal from stationary wavelet transform.

    Parameters
    ----------
    signal : float[n]
        Signal with spurious samples.

    Returns
    -------
    bool[n]
        Mask of valid samples.
    """
    # FIXME: Experimental. Does not work yet.

    import pywt
    wavelet = pywt.Wavelet("db9")
    signal_valid = np.copy(signal)

    power2_length = 1 + int(np.ceil(np.log2(len(signal_valid))))
    length_padded = 2 ** power2_length
    pad_width = length_padded - len(signal_valid)
    pad_before = pad_width // 2
    pad_after = pad_width - pad_before
    # mode = "constant"
    mode = "symmetric"
    signal_padded = pywt.pad(signal_valid, (pad_before, pad_after), mode)
    mask_padding = np.full(length_padded, np.nan)
    mask_padding[pad_before:length_padded - pad_after] = 1

    n_levels = 8
    normalise = True
    coef = pywt.swt(signal_padded, wavelet, level=n_levels, norm=normalise)

    def threshold_over(x, threshold=1):
        mask_over_threshold = np.abs(x) > threshold
        x_copy = x.copy()
        x_copy[mask_over_threshold] = 0
        return x_copy

    coef_thresholded = coef
    n_threshold_levels = 5
    threshold = 0.005
    for level in range(len(coef_thresholded))[-n_threshold_levels:]:
        ca, cd = coef_thresholded[level]
        cd_thresholded = threshold_over(cd, threshold=threshold)
        coef_thresholded[level] = ca, cd_thresholded
    for level in range(1, len(coef_thresholded)):
        ca, cd = coef_thresholded[level]
        ca_zeroed = np.zeros_like(ca)
        coef_thresholded[level] = ca_zeroed, cd
    signal_denoised = pywt.iswt(coef_thresholded, wavelet, norm=normalise)
    signal_denoised = signal_denoised[pad_before:length_padded - pad_after]
    assert len(signal_denoised) == len(signal_valid)

    return signal_denoised
