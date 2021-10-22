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


