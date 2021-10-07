# -*- coding: utf-8 -*-
"""Compute FDA-alpha1 index from reart rate variability (HRV).

This can be used for estimating the aerobic threshold.
"""
import argparse
import csv
import math

import fitparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine
import seaborn as sn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit")
    parser.add_argument("--cwt", action="store_true")
    parser.add_argument("--dfaa1", action="store_true")
    parser.add_argument("--dfaa1-motion", action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--pointcarre", action="store_true")
    parser.add_argument("--rr", action="store_true")
    parser.add_argument("--scatter", action="store_true")
    return parser.parse_args()


def load_rr_from_fit(path):
    fit_data = fitparse.FitFile(path)
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
    # window_size = 120
    window_size = 128
    step = 16
    index_start = 0
    # index_end = round(df["time"].max() / step)
    # index_end = int(np.floor((df["time"].max() - window_size) / step))
    # for index in range(index_start, index_end):
    index_end = int(np.floor((df["time"].max() - window_size)))
    for index in range(index_start, index_end, step):
        print(f"\rindex={int(index / step):03d}/{int(index_end / step):03d}",
              end="")
        window_mask = (
            (df['time'] >= index)
            & (df['time'] <= index + window_size)
        )
        array_rr = 1000 * df.loc[window_mask, "rr"]
        # compute heart rate
        heartrate = 60000 / np.mean(array_rr)
        nn_diff = np.abs(np.diff(array_rr))
        # rmssd = round(np.sqrt(np.sum((nn_diff * nn_diff) / len(nn_diff))), 2)
        rmssd = np.sqrt(np.mean((nn_diff ** 2)))
        sdnn = np.std(array_rr)
        alpha1 = compute_dfa(array_rr.to_list(), 4, 16)

        curr_features = {
            'time': index,
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


def main():
    args = parse_args()

    rr_raw = load_rr_from_fit(args.fit)
    mask_valid = compute_valid_mask(rr_raw)
    rr = rr_raw[mask_valid]
    time_ = np.cumsum(rr)

    if args.cwt:
        plot_cwt(rr_raw, mask_valid)

    if args.pointcarre:
        plot_pointcarre(rr_raw, mask_valid)

    if args.rr:
        plot_rr(rr_raw, mask_valid)

    if args.scatter:
        plot_scatter(rr_raw, mask_valid)

    df = pd.DataFrame()
    df["time"] = time_
    df["rr"] = rr

    # plot_df_rr(df)

    if args.dfaa1 or args.dfaa1_motion:
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


if __name__ == "__main__":
    main()
