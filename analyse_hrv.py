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
    parser.add_argument("--dfaa1", action="store_true")
    parser.add_argument("--dfaa1-motion", action="store_true")
    parser.add_argument("--dfaa1-vs-hr", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--pointcarre", action="store_true")
    parser.add_argument("--rr", action="store_true")
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--rr-cumsum", action="store_true")
    return parser.parse_args()


def compute_valid_mask(rr):
    """Find the valid samples in an RR interval sequence."""
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


def compute_dfa(pp_values, scale_min=4, scale_max=16, n_scales=None):
    if n_scales is None:
        n_scales = scale_max - scale_min + 1

    # Initialize. Using logarithmic scales.
    start = np.log(scale_min) / np.log(10)
    stop = np.log(scale_max) / np.log(10)
    scales = np.floor(np.logspace(start, stop, n_scales))
    F = np.zeros(n_scales)
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
        B = y.T
        x, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=None)

        errors = A @ x - B
        rmse_per_window = np.sqrt(np.mean(errors ** 2, axis=0))

        F[count] = np.sqrt(np.mean(rmse_per_window ** 2))
        count = count + 1

    pl2 = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = pl2[0]

    return alpha


def compute_features(df):
    features = []
    window_size = 2 ** 7
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
        rmssd = np.sqrt(np.mean((nn_diff ** 2)))
        sdnn = np.std(rr_window_s)
        alpha1 = compute_dfa(rr_window_s.copy(), 4, 16)

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
    mask_valid = compute_valid_mask(rr_raw)
    rr = rr_raw[mask_valid]
    time_ = np.cumsum(rr)

    if args.cwt:
        plot_cwt(rr_raw, mask_valid)

    if args.pointcarre:
        plot_pointcarre(rr_raw, mask_valid)

    if args.rr:
        plot_rr(rr_raw, mask_valid)

    if args.rr_cumsum:
        plot_rr_cumsum(rr_raw, mask_valid)

    if args.scatter:
        plot_scatter(rr_raw, mask_valid)

    df = pd.DataFrame()
    df["time"] = time_
    df["rr"] = rr

    if args.dfaa1 or args.dfaa1_motion or args.dfaa1_vs_hr:
        df_features = compute_features(df)

    if args.dfaa1:
        print(df_features.head())

        # Assuming a constant effort.
        mean_alpha1 = round(np.mean(df_features['alpha1']), 2)
        print(f"mean alpha1 = {mean_alpha1:.2f}")

        plot_df_alpha1(df_features)

    # Filter further based on SDNN to remove the moments standing still.

    if args.dfaa1_motion:
        # Based on visual inspection of the data.
        threshold_sdnn = 10

        mask_motion = df_features['sdnn'] < threshold_sdnn
        df_features_motion = df_features.loc[mask_motion]

        print(df_features_motion.head())

        mean_alpha1_motion = round(np.mean(df_features_motion['alpha1']), 2)
        print(f"mean alpha1 in motion = {mean_alpha1_motion:.2f}")

        plot_df_alpha1(df_features_motion)

    if args.dfaa1_vs_hr:
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
    plot_dfaa1, = ax.plot(time, alpha1, color=color_alpha1)
    ax.scatter(time, alpha1, c=alpha1, norm=color_normalizer, cmap=cmap)
    ax.set_xlabel("time")
    ax.set_ylabel("DFA-alpha1")
    ax.set_ylim((0, 1.5))
    ax.yaxis.label.set_color(plot_dfaa1.get_color())
    ax.tick_params(axis="y", colors=plot_dfaa1.get_color())
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
