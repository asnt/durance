# -*- coding: utf-8 -*-
"""Compute FDA-alpha1 index from reart rate variability (HRV).

This can be used for estimating the aerobic threshold.
"""
import argparse
import csv
import math
import pathlib

import fitparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--cwt", action="store_true")
    parser.add_argument("--swt", action="store_true")
    parser.add_argument("--dfa1", action="store_true")
    parser.add_argument("--dfa1-motion", action="store_true")
    parser.add_argument("--dfa1-vs-hr", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--lines", action="store_true")
    parser.add_argument("--pointcarre", action="store_true")
    parser.add_argument("--rr", action="store_true")
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--rr-cumsum", action="store_true")
    parser.add_argument("--outlier-method",
                        default="moving_median",
                        choices=["deviation", "moving_median", "wavelet"])
    return parser.parse_args()


def find_valid_deviation(rr):
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


def find_valid_moving_median(rr, window_size=31):
    """Find valid samples in an RR signal using a moving median.

    Parameters
    ----------
    rr: array-like
        (n,) signal of RR intervals.

    Returns
    -------
    mask_valid: array-like
        (n,) boolen mask array of the valid samples.
    """
    mask_valid = find_valid_deviation(rr)

    pad_before = window_size // 2
    pad_after = window_size - pad_before - 1
    pad_widths = pad_before, pad_after
    rr_padded = np.pad(rr, pad_widths,
                       mode="constant", constant_values=np.nan)

    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    windows = sliding_window_view(rr_padded, window_size)

    # XXX: np.nanmedian does not seem to work on a view of sliding-windows.
    # medians = np.nanmedian(windows, axis=1)
    medians = np.median(windows, axis=1)

    deviations = np.abs(rr - medians)

    # Determine the threshold for outliers.
    # 1) From statistics on the whole signal.
    #    For Polar H10.
    threshold = np.quantile(deviations, 0.80)
    #    For Garmin HRM-Dual.
    # threshold = np.quantile(deviations, 0.90)
    # 2) From statistics on the past window.
    #    XXX: Does not seem to work as well a the global statistics on a single
    #    example 20211011-run-easy.
    # windows_deviations = np.abs(windows - medians[:, None])
    # threshold = np.quantile(windows_deviations, 0.8, axis=1)

    mask_valid = deviations < threshold

    return mask_valid


def compute_dfa(pp_values, scale_min=16, scale_max=None, n_scales=None):
# XXX: Values that seem to work with Garmin HRM-dual. Using a window size of
# 2**8 or 2**9. Values different than Polar H10 because of different sampling
# rates?
# def compute_dfa(pp_values, scale_min=4, scale_max=16, n_scales=None):
    if scale_max is None:
        scale_max = int(len(pp_values) / 8)
    if n_scales is None:
        n_scales = scale_max - scale_min + 1
        n_scales = min(8, n_scales)
        # XXX: On 20211002-run-easy, setting n_scales=2 is similar to higher
        # number of scales, at least until 16.
        # n_scales = 2
    assert scale_min < scale_max

    start = np.log(scale_min) / np.log(10)
    stop = np.log(scale_max) / np.log(10)
    scales = np.floor(np.logspace(start, stop, n_scales))

    fluctuations = np.zeros(n_scales)
    count = 0

    pp = pp_values
    y_n = np.cumsum(pp - np.mean(pp))

    for scale in scales:
        width = int(scale)

        sliding_window_view = np.lib.stride_tricks.sliding_window_view
        y = sliding_window_view(y_n, width)

        A0 = np.arange(0, width).reshape(-1, 1)
        ones = np.ones((len(A0), 1))
        A = np.hstack((A0, ones))
        # TODO: Detrending with second-order polynomial. Seems to work with
        # longer window size and larger maximum scale (i.e. len(rr) / 4).
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


def compute_features(df):
    features = []
    window_size = 2 ** 8
    step = 16

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
        alpha1 = compute_dfa(rr_window_s.copy())

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


def plot_rr(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(rr)), rr, c=rr, cmap=cmap)


def plot_rr_cumsum(rr, mask_valid, cmap="hsv"):
    # rr_valid = rr[mask_valid]
    # rr_cumsum = np.cumsum(rr_valid - np.mean(rr_valid))
    rr_cumsum = np.cumsum(rr - np.mean(rr))
    fig, ax = plt.subplots()
    # ax.scatter(np.arange(len(rr_cumsum)), rr_cumsum)#, c=rr, cmap=cmap)
    ax.plot(rr_cumsum)


def plot_pointcarre(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    ax.scatter(rr[:-1], rr[1:], c=rr[:-1], cmap=cmap)


def plot_scatter(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    rr_valid = np.copy(rr)
    rr_valid[~mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_valid)
    rr_invalid = np.copy(rr)
    rr_invalid[mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_invalid)


def plot_lines(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    rr_valid = np.copy(rr)
    rr_valid[~mask_valid] = np.nan
    ax.plot(np.arange(len(rr)), rr_valid, color="black")
    rr_invalid = np.copy(rr)
    rr_invalid[mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_invalid, color="red", alpha=0.25)


def plot_cwt(rr, mask_valid, cmap="hsv"):
    import pywt
    rr_valid = np.copy(rr)
    rr_valid[~mask_valid] = np.nan
    scales = np.linspace(0.1, 8, 1024)
    coef, freq = pywt.cwt(rr_valid, scales, "mexh")
    plt.subplots()
    plt.matshow(coef)
    plt.subplots()
    plt.imshow(coef, cmap='PRGn', aspect='auto',
               vmax=abs(coef).max(), vmin=-abs(coef).max())


def plot_swt(rr, mask_valid, cmap="hsv"):
    import pywt
    wavelet = pywt.Wavelet("db9")
    rr_valid = np.copy(rr)
    # rr_valid[~mask_valid] = np.nan

    power2_length = 1 + int(np.ceil(np.log2(len(rr_valid))))
    length_padded = 2 ** power2_length
    # # rr_padded = np.full(length_padded, np.nan)
    # rr_padded = np.zeros(length_padded)
    # rr_padded[:len(rr_valid)] = rr_valid
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

    n_curves = 1 + 2 * len(coef)

    fig, axes = plt.subplots(nrows=n_curves)
    axes[0].plot(rr_padded)
    for index, (ca, cd) in enumerate(reversed(coef)):
        axes[1 + 2 * index].plot(ca)
        axes[1 + 2 * index + 1].plot(cd, alpha=0.25)

    def threshold_over(x, threshold=1):
        mask_over_threshold = np.abs(x) > threshold
        x_copy = x.copy()
        x_copy[mask_over_threshold] = 0
        return x_copy

    def threshold_soft(x, threshold=1):
        return np.sign(x) * np.maximum(0, np.abs(x) - threshold)

    coef_thresholded = coef
    n_threshold_levels = 5
    threshold = 0.005
    # threshold = 0.1
    for level in range(len(coef_thresholded))[-n_threshold_levels:]:
        ca, cd = coef_thresholded[level]
        # ca_thresholded = np.zeros_like(ca)
        # cd_thresholded = np.zeros_like(cd)
        ca_thresholded = threshold_over(ca, threshold=threshold)
        cd_thresholded = threshold_over(cd, threshold=threshold)
        coef_thresholded[level] = ca_thresholded, cd_thresholded
    for level in range(1, len(coef_thresholded)):
        ca, cd = coef_thresholded[level]
        ca_zeroed = np.zeros_like(ca)
        coef_thresholded[level] = ca_zeroed, cd
    rr_denoised = pywt.iswt(coef_thresholded, wavelet, norm=normalise)

    fig, axes = plt.subplots(nrows=n_curves)
    axes[0].plot(rr_padded)
    for index, (ca, cd) in enumerate(reversed(coef_thresholded)):
        axes[1 + 2 * index].plot(ca)
        axes[1 + 2 * index + 1].plot(cd, alpha=0.25)

    fig, axes = plt.subplots(nrows=3)
    x = np.arange(len(rr_padded))
    axes[0].plot(x, rr_padded)
    axes[1].plot(x, rr_denoised)
    axes[2].plot(x, rr_padded, alpha=0.25)
    axes[2].plot(x, rr_denoised)
    for ax in axes:
        ax.set_xlim(x[[pad_before, length_padded - pad_after]])


def denoise_swt(rr):
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


def load_rr_from_fit(path):
    fit_data = fitparse.FitFile(str(path))
    rr = []
    records = fit_data.get_messages('hrv')
    rr_intervals_with_nones = [
        record_data.value
        for record in records
        for record_data in record
    ]
    rr = np.array(rr_intervals_with_nones)
    rr = rr.flatten()
    rr = rr[rr != None]
    return rr


def load_rr_from_csv(path):
    return np.loadtxt(path)


def load_rr(path):
    if path.suffix == ".fit":
        rr_raw = load_rr_from_fit(path)
    elif path.suffix == ".csv":
        rr_raw_ms = load_rr_from_csv(path)
        rr_raw_s = rr_raw_ms / 1000
        rr_raw = rr_raw_s
    else:
        raise ValueError("input file not supported (must be .fit or .csv)")
    return rr_raw


def main():
    args = parse_args()

    rr_raw = load_rr(args.input)

    if args.outlier_method == "deviation":
        mask_valid = find_valid_deviation(rr_raw)
        rr = rr_raw[mask_valid]
    elif args.outlier_method == "moving_median":
        mask_valid = find_valid_moving_median(rr_raw)
        rr = rr_raw[mask_valid]
    elif args.outlier_method == "wavelet":
        # XXX: Does not work. Loss of details?
        mask_valid = np.full_like(rr_raw, True)
        rr = denoise_swt(rr_raw)

    time_ = np.cumsum(rr)

    if args.cwt:
        plot_cwt(rr_raw, mask_valid)

    if args.swt:
        plot_swt(rr_raw, mask_valid)

    if args.pointcarre:
        plot_pointcarre(rr_raw, mask_valid)

    if args.rr:
        plot_rr(rr_raw, mask_valid)

    if args.rr_cumsum:
        plot_rr_cumsum(rr_raw, mask_valid)

    if args.scatter:
        plot_scatter(rr_raw, mask_valid)

    if args.lines:
        plot_lines(rr_raw, mask_valid)

    df = pd.DataFrame()
    df["time"] = time_
    df["rr"] = rr

    if args.dfa1 or args.dfa1_motion or args.dfa1_vs_hr:
        df_features = compute_features(df)

    if args.dfa1:
        print(df_features.head())

        # Assuming a constant effort.
        mean_alpha1 = round(np.mean(df_features['alpha1']), 2)
        print(f"mean alpha1 = {mean_alpha1:.2f}")

        plot_df_alpha1(df_features)

    # Filter further based on SDNN to remove the moments standing still.

    if args.dfa1_motion:
        # Based on visual inspection of the data.
        threshold_sdnn = 10

        mask_motion = df_features['sdnn'] < threshold_sdnn
        df_features_motion = df_features.loc[mask_motion]

        print(df_features_motion.head())

        mean_alpha1_motion = round(np.mean(df_features_motion['alpha1']), 2)
        print(f"mean alpha1 in motion = {mean_alpha1_motion:.2f}")

        plot_df_alpha1(df_features_motion)

    if args.dfa1_vs_hr:
        plot_df_alpha1_vs_hr(df_features)

    plt.show()


def plot_df_alpha1(df, cmap="Spectral"):
    thresholds = [0.5, 0.75, 1.0]
    color_normalizer = mpl.colors.Normalize(vmin=thresholds[0],
                                            vmax=thresholds[-1])

    fig, ax = plt.subplots()

    time = df["time"].values
    alpha1 = df["alpha1"].values
    color_alpha1 = "dimgray"
    plot_dfa1, = ax.plot(time, alpha1, color=color_alpha1)
    ax.scatter(time, alpha1, c=alpha1, norm=color_normalizer, cmap=cmap)
    ax.set_xlabel("time")
    ax.set_ylabel("DFA-alpha1")
    ax.set_ylim((0, 1.5))
    ax.yaxis.label.set_color(plot_dfa1.get_color())
    ax.tick_params(axis="y", colors=plot_dfa1.get_color())
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(thresholds))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=0.1))
    ax.yaxis.grid(which="major", color="lightgray")

    ax_hr = ax.twinx()
    hr = df["heartrate"].values
    color_hr = "orangered"
    plot_hr, = ax_hr.plot(time, hr, color=color_hr, alpha=0.25)
    ax_hr.set_ylabel("heartrate")
    ax_hr.yaxis.label.set_color(plot_hr.get_color())
    ax_hr.tick_params(axis="y", colors=plot_hr.get_color())


def plot_df_alpha1_vs_hr(df, cmap="Spectral"):
    thresholds = [0.5, 0.75, 1.0]
    color_normalizer = mpl.colors.Normalize(vmin=thresholds[0],
                                            vmax=thresholds[-1])

    fig, ax = plt.subplots()

    hr = df["heartrate"].values
    alpha1 = df["alpha1"].values
    ax.scatter(hr, alpha1, c=alpha1, norm=color_normalizer, cmap=cmap)
    ax.set_xlabel("heartrate")
    ax.set_ylabel("DFA-alpha1")
    ax.set_ylim((0, 1.5))
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(thresholds))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=0.1))
    ax.yaxis.grid(which="major", color="lightgray")


if __name__ == "__main__":
    main()
