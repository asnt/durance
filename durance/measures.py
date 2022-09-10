from typing import Dict

import numpy as np


def dfa(
    signal: np.ndarray,
    scale_min: int = 16,
    scale_max: int = 64,
    n_scales_max: int = 16,
) -> float:
    """Estimate self-correlations with detrended fluctuation analysis (DFA).

    Detrend with polynomials of order 1 (DFA1).

    Returned exponent alpha > 0:
    - alpha < 0.5: anti-correlated
    - alpha = 0.5: uncorrelated (white noise)
    - alpha > 0.5: correlated
    - alpha = 1.0: 1/f-noise (pink noise)
    - alpha > 1.0: non-stationary, unbounded
    - alpha = 1.5: 1/f^2-noise (Brownian noise)

    Parameters
    ----------
    signal : float[n]
    scale_min :
        Minimum scale, i.e. smallest window width for detrending.
    scale_max :
        Maximum scale, i.e. largest window width for detrending.
    n_scales_max :
        Maximum number of intermediate scales between `scale_min` and
        `scale_max`.

    Returns
    -------
    alpha
        Scaling exponent indicating self-correlations of signal.

    References
    ----------
    .. [1] Peng, C-K., et al. "Mosaic organization of DNA nucleotides."
       Physical review e 49.2 (1994): 1685.
    .. [2] https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis
    """
    # Steps:
    # - Detrend at different scales:
    #     - set width of sub-window: width := scale (= time period),
    #     - compute trend in each (non-overlapping) sub-window (i.e. fit
    #       polynomial of given order),
    #     - compute residuals from trend in each sub-window.
    # - Compute fluctuation at each scale, i.e. root-mean-square of residuals
    #   from all sub-windows at given scale.
    # - Estimate exponent alpha of observed fluctutations, assuming power-law
    #   distribution of fluctuations across scales, i.e. fit function
    #       f(scale; alpha) = scale^alpha
    #   to observations
    #       f_k = f[scale_k],
    #   but in log space, to transform problem into linear regression,
    #       log(f(scale; alpha)) = log(scale^alpha) = alpha log(scale).
    # - Alpha indicates self-correlations.

    assert scale_min < scale_max
    assert n_scales_max > 2
    # TODO: Check how longer than largest window (scale_max) signal needs to be
    # for computation to make sense.
    assert len(signal) > 2 * scale_max

    n_scales = scale_max - scale_min + 1
    if n_scales_max is not None:
        n_scales = min(n_scales_max, n_scales)

    start = np.log2(scale_min)
    stop = np.log2(scale_max)
    scales = np.logspace(start, stop, n_scales, base=2)
    scales = np.round(scales).astype(int)

    fluctuations = np.zeros(n_scales)
    count = 0

    cumulated_signal = np.cumsum(signal - np.mean(signal))

    for scale in scales:
        width = scale

        sliding_window_view = np.lib.stride_tricks.sliding_window_view
        y = sliding_window_view(cumulated_signal, width)

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
        rmse = np.sqrt(np.mean(errors**2, axis=None))
        fluctuations[count] = rmse
        count = count + 1

    coeffs = np.polyfit(np.log(scales), np.log(fluctuations), 1)
    alpha = coeffs[0]

    return alpha


def dfa_batch(
    signal: np.ndarray,
    window_size: int = 2**8,
    scale_min: int = 16,
    scale_max: int = 64,
    n_scales_max: int = 16,
) -> np.ndarray:
    """Estimate self-correlations in windows around each point of a signal.

    Accelerate computation of DFA by dentrending once in all sliding windows of
    stride one at all scales.

    Parameters
    ----------
    signal : float[n]
    scale_min :
    scale_max :
    n_scales_max :

    Returns
    -------
    float[n]
        Exponents alpha indicating self-correlations.
    """
    assert scale_min < scale_max

    n_scales = scale_max - scale_min + 1
    if n_scales_max is not None:
        n_scales = min(n_scales_max, n_scales)

    start = np.log2(scale_min)
    stop = np.log2(scale_max)
    scales = np.logspace(start, stop, n_scales, base=2)
    scales = np.round(scales).astype(int)

    cumulated_signal = np.cumsum(signal - np.mean(signal))

    sliding_window_view = np.lib.stride_tricks.sliding_window_view

    errors_per_scale = []
    for scale in scales:
        width = scale

        y = sliding_window_view(cumulated_signal, width)
        # Note: sliding_window_view outputs dtype=object. Need explicit casting
        # to dtype=float for operations below to work, e.g. np.nanmean.
        y = y.astype(float)

        A0 = np.arange(width).reshape(-1, 1)
        ones = np.ones((len(A0), 1))
        A = np.hstack((A0, ones))
        B = y.T
        x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        errors = A @ x - B
        errors_per_scale.append(errors.T)

    errors_windows = [
        sliding_window_view(errors, window_size, axis=0, subok=True)
        for errors in errors_per_scale
    ]

    fluctuations_per_scale = [
        np.sqrt(np.nanmean(errors**2, axis=(1, 2)))
        for errors in errors_windows
    ]

    # Crop signals symmetrically to length of smallest signal (largest window
    # size).
    min_length = len(fluctuations_per_scale[-1])
    crop_widths = []
    for array in fluctuations_per_scale:
        length = len(array)
        excess = length - min_length
        crop_left = excess // 2
        crop_right = excess - crop_left
        crop_widths.append((crop_left, crop_right))
    fluctuations_per_scale = [
        array[crop[0]:-crop[1]] if crop[1] > 0 else array[crop[0]:]
        for crop, array in zip(crop_widths, fluctuations_per_scale)
    ]

    log2_scales = np.log2(scales)
    log2_fluctuations = np.log2(fluctuations_per_scale)

    B = log2_fluctuations
    ones = np.ones((len(log2_scales), 1))
    A = np.hstack((log2_scales[:, None], ones))
    x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    alphas = x[0]

    return alphas


def features_from_sliding_window(
        hrv_signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute HRV features in sliding window over HRV signal.

    Parameters
    ----------
    hrv_signals
        rr : float[n]
            RR signal, i.e. heartbeat intervals (seconds).
        datetime : datetime[n]
            Datetime of samples.

    Returns
    -------
    hrv_features

        Features from k overlapping windows over the input signal.

        index : int[k]
        datetime : datetime[k]
        heartrate : float[k]
        rmssd : float[k]
        sdnn : float[k]
        alpha1 : float[k]
    """
    features = []
    window_size = 2**8
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
        rmssd = np.sqrt(np.mean(nn_diff**2))
        sdnn = np.std(rr_window_s)
        alpha1 = dfa(rr_window_s.copy(), n_scales_max=n_scales_max)

        curr_features = {
            "index": index,
            "datetime": datetime[index],
            "heartrate": heartrate,
            "rmssd": rmssd,
            "sdnn": sdnn,
            "alpha1": alpha1,
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
    features = {name: np.stack(values) for name, values in features.items()}

    return features


def _pad_like(
    a: np.ndarray,
    b: np.ndarray,
    align: str = "center",
) -> np.ndarray:
    """Pad `a` with NaN's to length of `b`, with horizontal alignment.

    If `len(a) >= len(b)`, return `a` unmodified.

    Parameters
    ----------
    a : Any[n]
    b : Any[m]
    align : {"center"}
        Alignment of input array into padded array.
    """
    # TODO: Implement left and right alignment whenever needed.
    assert align == "center"

    gap = len(b) - len(a)
    if gap <= 0:
        # Cannot pad: `a` longer than or the same size as `b`.
        return a
    pad_left = int(gap / 2)
    pad_right = gap - pad_left
    pad_width = (pad_left, pad_right)
    return np.pad(a, pad_width=pad_width, constant_values=np.nan)


def features_from_sliding_window_2(
        hrv_signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute HRV features in sliding window over HRV signal.

    Parameters
    ----------
    hrv_signals
        rr : float[n]
            RR signal, i.e. duration of intervals between heartbeats.
        datetime : datetime[n]
            Datetime of samples.
        relative_time_s : float[n]
            Relative time (seconds) since first sample.

    Returns
    -------
    hrv_features
    """
    features = []
    window_size = 2**8
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
    rmssd = np.sqrt(np.mean(nn_diff**2, axis=1))
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
