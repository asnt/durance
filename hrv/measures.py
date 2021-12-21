import numpy as np
import pandas as pd


def dfa(pp, scale_min=16, scale_max=32, n_scales_max=None):
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
        # Zero-th order polynomial. The same as centred moving average (CMA)
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


def dfa_batch(rr, window_size=2 ** 8, scale_min=16, scale_max=32,
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


def features_from_sliding_window(df):
    features = []
    window_size = 2 ** 8
    step = 16
    n_scales_max = 16

    rr = df["rr"].values
    times = df["time"].values

    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    rr_windows_ms = sliding_window_view(rr, window_size)

    rr_windows_ms = rr_windows_ms[::step]
    times = times[round(step // 2)::step]

    n_windows = len(rr_windows_ms)
    for index, rr_window_ms in enumerate(rr_windows_ms):
        print(f"\rindex={index:03d}/{n_windows:03d}", end="")

        rr_window_s = rr_window_ms * 1000
        rr_window_s = list(rr_window_s)

        heartrate = 60000 / np.mean(rr_window_s)
        nn_diff = np.abs(np.diff(rr_window_s))
        rmssd = np.sqrt(np.mean(nn_diff ** 2))
        sdnn = np.std(rr_window_s)
        alpha1 = dfa(rr_window_s.copy(), n_scales_max=n_scales_max)

        curr_features = {
            'index': index,
            'time': times[index],
            'heartrate': heartrate,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'alpha1': alpha1,
        }

        features.append(curr_features)
    print()

    df_features = pd.DataFrame(features)

    return df_features


def features_from_sliding_window_2(df):
    features = []
    window_size = 2 ** 8
    step = 1
    n_scales_max = 16

    rr_s = df["rr"].values
    times = df["time"].values

    rr_ms = rr_s * 1000
    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    rr_windows = sliding_window_view(rr_ms, window_size)
    rr_windows = rr_windows.astype(float)

    rr_windows = rr_windows[::step]
    times = times[:-window_size:step]

    heartrate = 60_000 / np.mean(rr_windows, axis=1)
    nn_diff = np.abs(np.diff(rr_windows, axis=1))
    rmssd = np.sqrt(np.mean(nn_diff ** 2, axis=1))
    sdnn = np.std(rr_windows, axis=1)

    rr = df["rr"]
    alpha1 = dfa_batch(rr, n_scales_max=n_scales_max)
    n_samples = min(len(alpha1), len(times))

    times = times[:n_samples]
    heartrate = heartrate[:n_samples]
    rmssd = rmssd[:n_samples]
    sdnn = sdnn[:n_samples]
    heartrate = heartrate[:n_samples]
    alpha1 = alpha1[:n_samples]

    features = {
        "index": np.arange(n_samples, dtype=int),
        "time": times,
        "heartrate": heartrate,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "alpha1": alpha1,
    }

    df_features = pd.DataFrame(features)

    return df_features
