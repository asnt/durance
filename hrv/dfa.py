import numpy as np


def compute_dfa(pp, scale_min=16, scale_max=32, n_scales_max=None):
    assert scale_min < scale_max

    n_scales = scale_max - scale_min + 1
    if n_scales_max is not None:
        n_scales = min(n_scales_max, n_scales)

    start = np.log(scale_min) / np.log(10)
    stop = np.log(scale_max) / np.log(10)
    scales = np.logspace(start, stop, n_scales)
    scales = np.round(scales).astype(int)

    fluctuations = np.zeros(n_scales)
    count = 0

    y_n = np.cumsum(pp - np.mean(pp))

    for scale in scales:
        width = scale

        sliding_window_view = np.lib.stride_tricks.sliding_window_view
        y = sliding_window_view(y_n, width)

        A0 = np.arange(0, width).reshape(-1, 1)
        ones = np.ones((len(A0), 1))
        # Zero-th order polynomial. The same a centred moving average (CMA)
        # method.
        # A = ones
        # First order polynomial
        A = np.hstack((A0, ones))
        # Second order polynomial
        # TODO: Check how this compares to a first-order polynomial. Seems to
        # work with longer window size and larger maximum scale
        # (i.e. len(rr) / 4).
        # A = np.hstack((A0 ** 2, A0, ones))
        B = y.T
        x, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=None)

        errors = A @ x - B
        rmse = np.sqrt(np.mean(errors ** 2, axis=None))
        fluctuations[count] = rmse
        count = count + 1

    coeffs = np.polyfit(np.log2(scales), np.log2(fluctuations), 1)
    alpha = coeffs[0]

    return alpha


def compute_dfa_batch(rr,
                      window_size=2 ** 8,
                      scale_min=16,
                      scale_max=32,
                      n_scales_max=None):
    assert scale_min < scale_max

    n_scales = scale_max - scale_min + 1
    if n_scales_max is not None:
        n_scales = min(n_scales_max, n_scales)

    start = np.log(scale_min) / np.log(10)
    stop = np.log(scale_max) / np.log(10)
    scales = np.logspace(start, stop, n_scales)
    scales = np.round(scales).astype(int)

    y_n = np.cumsum(rr - np.mean(rr))

    errors_per_scale = []
    for scale in scales:
        width = scale

        sliding_window_view = np.lib.stride_tricks.sliding_window_view
        y = sliding_window_view(y_n, width)
        y = y.astype(float)

        A0 = np.arange(width).reshape(-1, 1)
        ones = np.ones((len(A0), 1))
        A = np.hstack((A0, ones))
        B = y.T
        x, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=None)

        errors = A @ x - B
        errors_per_scale.append(errors.T)

    errors_windows = [
        sliding_window_view(errors, window_size, axis=0, subok=True)
        for errors in errors_per_scale
    ]

    fluctuations_per_scale = [
        np.sqrt(np.nanmean(errors ** 2, axis=(1, 2)))
        for errors in errors_windows
    ]

    n_samples_min = np.amin([
        fluctuations.shape[0]
        for fluctuations in fluctuations_per_scale
    ])

    fluctuations_per_scale = np.stack([
        fluctuations[:n_samples_min]
        for fluctuations in fluctuations_per_scale
    ])

    log2_scales = np.log2(scales)
    log2_fluctuations = np.log2(fluctuations_per_scale)

    B = log2_fluctuations
    ones = np.ones((len(log2_scales), 1))
    A = np.hstack((log2_scales[:, None], ones))
    x, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=None)

    alpha = x[0]

    return alpha
