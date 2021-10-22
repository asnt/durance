import numpy as np


def inliers_from_deviation(rr):
    """Find the valid samples in an RR signal from sample-to-sample deviation.

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


def inliers_from_moving_median(rr, window_size=31):
    """Find valid samples in an RR signal using a moving median.

    Parameters
    ----------
    rr: array-like
        (n,) signal of RR intervals.
    window_size: int
        Size of the moving window.

    Returns
    -------
    mask_valid: array-like
        (n,) boolen mask array of the valid samples.
    """
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

    # Determine the threshold for outliers.
    # 1) From statistics on the whole signal.
    #    For Polar H10.
    # threshold = np.quantile(deviations, 0.80)
    threshold = np.quantile(deviations, 0.85)
    #    For Garmin HRM-Dual.
    # threshold = np.quantile(deviations, 0.90)
    # 2) From statistics on the past window.
    #    XXX: Does not seem to work as well as the global statistics on a
    #    single example 20211011-run-easy.
    # windows_deviations = np.abs(windows - medians[:, None])
    # threshold = np.quantile(windows_deviations, 0.8, axis=1)

    mask_valid = deviations < threshold

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
