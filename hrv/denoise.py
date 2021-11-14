import numpy as np


def find_inliers(rr, method="moving_median"):
    """Find valid RR signal samples.

    Parameters
    ----------
    rr: array-like
        (n,) Raw signal of RR intervals.
    method: str, {"moving_median", "deviation", "deviation_forward"}
        The method to identify outliers.

    Returns
    -------
    mask_valid: array-like
        (n,) boolen mask array of the valid samples.
    """
    if method == "deviation":
        mask_valid = inliers_from_deviation(rr)
    elif method == "deviation_forward":
        mask_valid = inliers_from_deviation_forward(rr)
    elif method == "moving_median":
        mask_valid = inliers_from_moving_median(rr)
    elif method == "wavelet":
        raise NotImplementedError
        # XXX: Does not work. Loss of details?
        mask_valid = np.full_like(rr, True)
    else:
        raise ValueError(f"invalid method {method}")
    return mask_valid


def inliers_from_deviation_forward(rr):
    """Find valid RR signal samples from the deviation to the next sample.

    Parameters
    ----------
    rr: array-like
        (n,) signal of RR intervals.

    Returns
    -------
    mask_valid: array-like
        (n,) boolen mask array of the valid samples.
    """
    diff_0 = np.diff(rr)
    diff_0 = np.concatenate((diff_0, [np.inf]))
    relative_variation_0 = np.abs(diff_0) / np.abs(rr)
    threshold_variation = 0.05
    mask_valid = relative_variation_0 < threshold_variation
    return mask_valid


def inliers_from_deviation(rr):
    """Find valid RR signal samples from sample-to-sample deviation.

    The deviations from both the previous and the next samples must be within
    the threshold.

    Parameters
    ----------
    rr: array-like
        (n,) signal of RR intervals.

    Returns
    -------
    mask_valid: array-like
        (n,) boolen mask array of the valid samples.
    """
    diff_0 = np.diff(rr)
    diff_1 = np.diff(rr[::-1])[::-1]
    diff_0 = np.concatenate((diff_0, [0]))
    diff_1 = np.concatenate(([0], diff_1))
    relative_variation_0 = np.abs(diff_0) / np.abs(rr)
    relative_variation_1 = np.abs(diff_1) / np.abs(rr)
    threshold_variation = 0.05
    # threshold_variation = 0.10
    mask_valid = (
        (relative_variation_0 < threshold_variation)
        & (relative_variation_1 < threshold_variation)
    )
    return mask_valid


def inliers_from_moving_median(
    rr: np.ndarray,
    window_size: int = 31,
    method: str = "absolute",
    threshold: float = 0.015,
) -> np.ndarray:
    """Find valid samples in an RR signal using a moving median.

    Parameters
    ----------
    rr: array-like
        (n,) signal of RR intervals.
    window_size: int
        Size of the moving window.
    method: {"absolute", "quantile"}
        Method to determine inliers.
    threshold: float
        Maxixum distance from the moving median for an inlier. Maximum
        deviation in seconds for method 'absolute' (e.g. 0.015). Quantile value
        (unitless) in (0, 1) for method 'quantile' (e.g. 0.85).

    Returns
    -------
    mask_valid: array-like
        (n,) boolen mask array of the valid samples.
    """
    assert method in ("absolute", "quantile")
    if method == "quantile":
        assert 0 < threshold < 1.0

    pad_before = window_size // 2
    pad_after = window_size - pad_before - 1
    pad_widths = pad_before, pad_after
    rr_padded = np.pad(rr, pad_widths,
                       mode="constant", constant_values=np.nan)

    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    windows = sliding_window_view(rr_padded, window_size)
    # Need explicit cast to dtype float for some operations below (e.g.
    # np.nanmedian). By default, the sliding window views have dtype object.
    windows = windows.astype(float)

    medians = np.nanmedian(windows, axis=1)
    deviations = np.abs(rr - medians)

    if method == "absolute":
        threshold_ = threshold
    elif method == "quantile":
        threshold_ = np.quantile(deviations, threshold)

    mask_valid = deviations < threshold_

    return mask_valid


def inliers_from_swt(rr):
    import pywt
    wavelet = pywt.Wavelet("db9")
    rr_valid = np.copy(rr)

    power2_length = 1 + int(np.ceil(np.log2(len(rr_valid))))
    length_padded = 2 ** power2_length
    pad_width = length_padded - len(rr_valid)
    pad_before = pad_width // 2
    pad_after = pad_width - pad_before
    # mode = "constant"
    mode = "symmetric"
    rr_padded = pywt.pad(rr_valid, (pad_before, pad_after), mode)
    mask_padding = np.full(length_padded, np.nan)
    mask_padding[pad_before:length_padded - pad_after] = 1

    n_levels = 8
    normalise = True
    coef = pywt.swt(rr_padded, wavelet, level=n_levels, norm=normalise)

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
    rr_denoised = pywt.iswt(coef_thresholded, wavelet, norm=normalise)
    rr_denoised = rr_denoised[pad_before:length_padded - pad_after]
    assert len(rr_denoised) == len(rr_valid)

    return rr_denoised
