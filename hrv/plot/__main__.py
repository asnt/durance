# -*- coding: utf-8 -*-
"""Compute DFA-alpha1 index from heart rate variability (HRV).

This can be used for estimating the aerobic threshold.
"""

import argparse
import importlib
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hrv.data
import hrv.denoise
import hrv.measures


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--backend", default="matplotlib",
                        choices=["bokeh", "matplotlib"])
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
                        choices=["deviation", "deviation_forward",
                                 "moving_median", "wavelet"])
    return parser.parse_args()


def main():
    args = parse_args()

    plot = importlib.import_module(f"hrv.plot.{args.backend}")

    rr_raw = hrv.data.load_rr(args.input)
    mask_valid = hrv.denoise.find_inliers(rr_raw, method=args.outlier_method)
    rr = rr_raw[mask_valid]

    if args.rr_average:
        rr_average = compute_moving_average(rr_raw, average_fn="median")

    n_valid = mask_valid.sum()
    n_samples = len(mask_valid)
    proportion_valid = n_valid / n_samples
    print(f"proportion valid = {proportion_valid:.2f}")

    time_relative = np.cumsum(rr)
    time_relative = time_relative.astype(float)

    activity_data, _ = hrv.data.load(args.input)
    datetime_start = np.datetime64(activity_data["datetime_start"])
    timedelta = time_relative.astype("timedelta64[s]")
    datetimes = datetime_start + timedelta
    time_ = datetimes

    if args.cwt:
        plot.cwt(rr_raw, mask_valid)

    if args.swt:
        plot.swt(rr_raw, mask_valid)

    if args.pointcarre:
        plot.pointcarre(rr_raw, mask_valid)

    if args.rr:
        plot.rr(rr_raw, mask_valid)

    if args.rr_cumsum:
        plot.rr_cumsum(rr_raw, mask_valid)

    if args.scatter:
        plot.scatter(rr_raw, mask_valid)

    if args.lines:
        plot.lines(rr_raw, mask_valid)

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
            df_features = hrv.measures.features_from_sliding_window(df)
        elif args.dfa1_mode == "batch":
            df_features = hrv.measures.features_from_sliding_window_2(df)

    if args.sdnn:
        sdnn = df_features["sdnn"]
        time = df_features["time"]
        fig, ax = plot.series(sdnn, x=time)
        fig.autofmt_xdate()
        ax.set_title("sdnn")
        ax.set_ylabel("sdnn")

    if args.rmssd:
        rmssd = df_features["rmssd"]
        time = df_features["time"]
        fig, ax = plot.series(rmssd, x=time)
        fig.autofmt_xdate()
        ax.set_title("rmssd")
        ax.set_ylabel("rmssd")

    if args.dfa1:
        print(df_features.head())

        # Assuming a constant effort.
        mean_alpha1 = round(np.mean(df_features['alpha1']), 2)
        print(f"mean alpha1 = {mean_alpha1:.2f}")

        plot.df_alpha1(df_features)

    # Filter further based on SDNN to remove the moments standing still.

    if args.dfa1_motion:
        # Based on visual inspection of the data.
        threshold_sdnn = 10

        mask_motion = df_features['sdnn'] < threshold_sdnn
        df_features_motion = df_features.loc[mask_motion]

        print(df_features_motion.head())

        mean_alpha1_motion = round(np.mean(df_features_motion['alpha1']), 2)
        print(f"mean alpha1 in motion = {mean_alpha1_motion:.2f}")

        plot.df_alpha1(df_features_motion)

    if args.dfa1_vs_hr:
        plot.df_alpha1_vs_hr(df_features)

    if args.overlay:
        plot.overlay(df_features)

    plot.show()


if __name__ == "__main__":
    main()
