# -*- coding: utf-8 -*-
"""Compute FDA-alpha1 index from reart rate variability (HRV).

This can be used for estimating the aerobic threshold.
"""
import argparse
import csv
import math
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import denoise
import features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--cwt", action="store_true")
    parser.add_argument("--swt", action="store_true")
    parser.add_argument("--dfa1-mode", default="per_window",
                        choices=["per_window", "batch"])
    parser.add_argument("--dfa1", action="store_true")
    parser.add_argument("--dfa1-motion", action="store_true")
    parser.add_argument("--dfa1-vs-hr", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--lines", action="store_true")
    parser.add_argument("--pointcarre", action="store_true")
    parser.add_argument("--rmssd", action="store_true")
    parser.add_argument("--rr", action="store_true")
    parser.add_argument("--rr-average", action="store_true")
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--sdnn", action="store_true")
    parser.add_argument("--rr-cumsum", action="store_true")
    parser.add_argument("--outlier-method",
                        default="moving_median",
                        choices=["deviation", "moving_median", "wavelet"])
    return parser.parse_args()


def compute_moving_average(x, window_size=31, average_fn="mean"):
    """Compute the moving average over a signal.

    Parameters
    ----------
    rr: array-like
        (n,) signal of RR intervals.
    window_size: int
        Size of the moving window.

    Returns
    -------
    average_signal: array-like
        (n,) moving average of the input signal x.
    """
    pad_before = window_size // 2
    pad_after = window_size - pad_before - 1
    pad_widths = pad_before, pad_after
    x_padded = np.pad(x, pad_widths, mode="constant", constant_values=np.nan)

    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    windows = sliding_window_view(x_padded, window_size)

    if average_fn == "mean":
        average_signal = np.nanmean(windows, axis=1)
    elif average_fn == "median":
        average_signal = np.quantile(windows, 0.5, axis=1)
        # average_signal = np.nanquantile(windows, 0.5, axis=1)

    return average_signal


def plot_rr(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(rr)), rr, c=rr, cmap=cmap)


def plot_rr_cumsum(rr, mask_valid, cmap="hsv"):
    rr_cumsum = np.cumsum(rr - np.mean(rr))
    fig, ax = plt.subplots()

    rr_valid = rr[mask_valid]
    rr_valid_cumsum = np.cumsum(rr_valid - np.mean(rr_valid))
    rr_valid_cumsum_masked = np.full_like(rr, np.nan)
    rr_valid_cumsum_masked[mask_valid] = rr_valid_cumsum

    x = np.arange(len(rr_cumsum))
    ax.plot(x, rr_cumsum, color="black", alpha=0.25, linewidth=2)
    ax.plot(x, rr_valid_cumsum_masked, color="black")


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


def plot_series(y, mask_valid=True, x=None, cmap="hsv"):
    fig, ax = plt.subplots()
    y_valid = np.copy(y)
    y_valid[~mask_valid] = np.nan
    if x is None:
        x = np.arange(len(y))
    ax.plot(x, y, color="black")
    return fig, ax


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


def main():
    args = parse_args()

    rr_raw = data.load_rr(args.input)

    if args.outlier_method == "deviation":
        mask_valid = denoise.inliers_from_deviation(rr_raw)
        rr = rr_raw[mask_valid]
    elif args.outlier_method == "moving_median":
        mask_valid = denoise.inliers_from_moving_median(rr_raw)
        rr = rr_raw[mask_valid]
    elif args.outlier_method == "wavelet":
        # XXX: Does not work. Loss of details?
        mask_valid = np.full_like(rr_raw, True)
        rr = denoise.inliers_from_swt(rr_raw)

    if args.rr_average:
        rr_average = compute_moving_average(rr_raw, average_fn="median")

    n_valid = mask_valid.sum()
    n_samples = len(mask_valid)
    proportion_valid = n_valid / n_samples
    print(f"proportion valid = {proportion_valid:.2f}")

    time_ = np.cumsum(rr)

    if args.input.suffix == ".fit":
        records = data.load_fit_records(args.input)
        dataframe = pd.DataFrame.from_records(records)
        start_timestamp_ = dataframe["timestamp"][0]
        start_datetime = start_timestamp_.to_pydatetime()
        start_timestamp = start_datetime.timestamp()
        timestamps = [
            start_timestamp + time__
            for time__ in time_
        ]

        import datetime

        datetimes = [
            datetime.datetime.fromtimestamp(timestamp)
            for timestamp in timestamps
        ]
        time_ = datetimes

        # TODO: Find a way to pass durations to matplotlib.
        # `datetime.timedelta`'s do not seem directly usable by matplotlib, it
        # seems.
        # timedeltas = [
        #     datetime.timedelta(seconds=timestamp)
        #     for timestamp in timestamps
        # ]
        # time_ = timedeltas

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

    if args.rr_average:
        fig, ax = plt.subplots()
        ax.scatter(np.arange(len(rr_raw)), rr_raw, alpha=0.25,
                   color="orangered")
        ax.plot(rr_average)

    df = pd.DataFrame()
    df["time"] = time_
    df["rr"] = rr
    if args.rr_average:
        df["rr_average"] = rr_average[mask_valid]

    require_features = (
        args.dfa1
        or args.dfa1_motion
        or args.dfa1_vs_hr
        or args.sdnn
        or args.rmssd
        or args.overlay
    )

    if require_features:
        if args.dfa1_mode == "per_window":
            df_features = features.compute_features(df)
        elif args.dfa1_mode == "batch":
            df_features = features.compute_features_2(df)

    if args.sdnn:
        sdnn = df_features["sdnn"]
        time = df_features["time"]
        fig, ax = plot_series(sdnn, x=time)
        fig.autofmt_xdate()
        ax.set_title("sdnn")
        ax.set_ylabel("sdnn")

    if args.rmssd:
        rmssd = df_features["rmssd"]
        time = df_features["time"]
        fig, ax = plot_series(rmssd, x=time)
        fig.autofmt_xdate()
        ax.set_title("rmssd")
        ax.set_ylabel("rmssd")

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

    if args.overlay:
        plot_overlay(df_features)

    plt.show()


def plot_df_alpha1(df, cmap="Spectral"):
    thresholds = [0.5, 0.75, 1.0]
    color_normalizer = mpl.colors.Normalize(vmin=thresholds[0],
                                            vmax=thresholds[-1])

    fig, ax = plt.subplots()

    time = df["time"].values
    # time = mpl.dates.num2timedelta(time)
    # time = mpl.dates.date2num(time)

    alpha1 = df["alpha1"].values
    color_alpha1 = "dimgray"
    plot_dfa1, = ax.plot(time, alpha1, color=color_alpha1)
    ax.scatter(time, alpha1, c=alpha1, norm=color_normalizer, cmap=cmap)
    ax.set_xlabel("time")
    ax.set_ylabel("DFA-alpha1")
    ax.set_ylim((0, 1.5))
    fig.autofmt_xdate()
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


def plot_overlay(df, cmap="Spectral"):
    dfa_thresholds = [0.5, 0.75, 1.0]
    color_normalizer = mpl.colors.Normalize(vmin=dfa_thresholds[0],
                                            vmax=dfa_thresholds[-1])
    dfa_ticks = dfa_thresholds + [1.5, 2.0]

    time = df["time"]
    alpha1 = df["alpha1"].values

    fig, ax = plt.subplots()

    time = df["time"].values

    alpha1 = df["alpha1"].values
    color_alpha1 = "dimgray"
    plot_dfa1, = ax.plot(time, alpha1, color=color_alpha1)
    ax.scatter(time, alpha1, c=alpha1, norm=color_normalizer, cmap=cmap)
    ax.set_xlabel("time")
    ax.set_ylabel("DFA1")
    ax.set_ylim((0, dfa_ticks[-1]))
    fig.autofmt_xdate()
    ax.yaxis.label.set_color(plot_dfa1.get_color())
    ax.tick_params(axis="y", colors=plot_dfa1.get_color())
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(dfa_ticks))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=0.1))
    ax.yaxis.grid(which="major", color="lightgray")

    params = dict(
        heartrate=dict(
            color="orangered",
            alpha=0.25,
        ),
        sdnn=dict(
            color="darkgray",
            linestyle="dotted",
        ),
        rmssd=dict(
            color="gray",
            linestyle="dashed",
        ),
    )

    for index, feature in enumerate(params):
        ax_ = ax.twinx()
        ax_.spines.right.set_position(("axes", 1 + index / 10))
        y = df[feature].values
        plot_, = ax_.plot(time, y, **params[feature])
        ax_.set_ylabel(feature)
        ax_.yaxis.label.set_color(plot_.get_color())
        ax_.tick_params(axis="y", colors=plot_.get_color())

    fig.tight_layout()


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
