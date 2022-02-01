import numpy as np
import pandas as pd


def dfa(pp, scale_min=16, scale_max=64, n_scales_max=16):
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


def dfa_batch(rr, window_size=2 ** 8, scale_min=16, scale_max=64,
              n_scales_max=16):
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


def features_from_sliding_window(hrv_signals):
    features = []
    window_size = 2 ** 8
    step = 16
    n_scales_max = 16

    rr = hrv_signals["rr"]
    datetime = hrv_signals["datetime"]

    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    rr_windows_ms = sliding_window_view(rr, window_size)

    rr_windows_ms = rr_windows_ms[::step]
    datetime = datetime[round(step // 2)::step]

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
            'datetime': datetime[index],
            'heartrate': heartrate,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'alpha1': alpha1,
        }

        features.append(curr_features)
    print()

    def zipdicts(*dicts):
        """Zip the values of the common keys of a sequence of dicts.

        Reference: https://stackoverflow.com/a/16458780
        """
        if not dicts:
            return
        common_keys = set(dicts[0]).intersection(*dicts[1:2])
        for key in common_keys:
            yield key, tuple(d[key] for d in dicts)

    features = dict(zipdicts(*features))
    features = {
        name: np.stack(values)
        for name, values in features.items()
    }

    return features


def _pad_like(a, b, align="center"):
    """Pad `a` to the length of `b` with NaN's and horizontal alignment.

    If `len(a) >= len(b)`, return `a` unmodified.
    """
    assert align == "center"
    gap = len(b) - len(a)
    if gap <= 0:
        # Cannot pad: `a` longer than or the same size as `b`.
        return a
    pad_left = int(gap / 2)
    pad_right = gap - pad_left
    pad_width = (pad_left, pad_right)
    return np.pad(a, pad_width=pad_width, constant_values=np.nan)


def features_from_sliding_window_2(hrv_signals):
    features = []
    window_size = 2 ** 8
    step = 1
    n_scales_max = 16

    # TODO: Ensure downsampling is handled correctly below when padding the
    # signals back to the original length.
    assert step == 1, "downsampling not yet supported"

    rr_s = hrv_signals["rr"]
    relative_time_s = hrv_signals["relative_time_s"]
    datetime = hrv_signals["datetime"]

    rr_ms = rr_s * 1000
    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    rr_windows = sliding_window_view(rr_ms, window_size)
    rr_windows = rr_windows.astype(float)

    if step > 1:
        rr_windows = rr_windows[::step]
        # TODO: Check this is correct.
        relative_time_s = relative_time_s[int(window_size / 2)::step]
        datetime = datetime[int(window_size / 2)::step]

    heartrate = 60_000 / np.mean(rr_windows, axis=1)
    nn_diff = np.abs(np.diff(rr_windows, axis=1))
    rmssd = np.sqrt(np.mean(nn_diff ** 2, axis=1))
    sdnn = np.std(rr_windows, axis=1)

    alpha1 = dfa_batch(rr_s, n_scales_max=n_scales_max)

    features = {
        "datetime": datetime,
        "relative_time_s": relative_time_s,
        "heartrate": heartrate,
        "rmssd": rmssd,
        "sdnn": sdnn,
        "alpha1": alpha1,
    }

    features = {
        name: _pad_like(signal, rr_ms, align="center")
        for name, signal in features.items()
    }
    features["index"] = np.arange(len(rr_ms), dtype=int)

    return features
