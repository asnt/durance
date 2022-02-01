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


def cleanup_rr_signal(rr_raw, outlier_method="moving_median"):
    mask_valid = hrv.denoise.find_inliers(rr_raw, method=outlier_method)
    rr = rr_raw[mask_valid]
    return rr, mask_valid


def main():
    args = parse_args()

    plot = importlib.import_module(f"hrv.plot.{args.backend}")

    activity_data, signals = hrv.data.load(args.input)

    rr_raw = hrv.data.load_rr(args.input)
    rr, mask_valid = cleanup_rr_signal(
        rr_raw,
        outlier_method=args.outlier_method,
    )
    hrv_relative_time_s = np.cumsum(rr_raw)[mask_valid].astype(float)

    n_valid = mask_valid.sum()
    n_samples = len(mask_valid)
    proportion_valid = n_valid / n_samples
    print(f"proportion valid = {proportion_valid:.2f}")

    if args.rr_average:
        rr_average = compute_moving_average(rr_raw, average_fn="median")

    activity_data, _ = hrv.data.load(args.input)

    timestamps_s = signals["timestamp"]
    timestamps_ms = 1000 * timestamps_s
    datetime = timestamps_ms.astype("datetime64[ms]")
    signals["datetime"] = datetime

    start_datetime = datetime[0:1]
    hrv_relative_time_ms = 1000 * hrv_relative_time_s
    hrv_timedelta = hrv_relative_time_ms.astype("timedelta64[ms]")
    hrv_datetime = start_datetime + hrv_timedelta

    hrv_signals = dict(
        datetime=hrv_datetime,
        relative_time_s=hrv_relative_time_s,
        rr=rr,
    )
    if args.rr_average:
        hrv_signals["rr_average"] = rr_average[mask_valid]

    require_features = (
        args.dfa1
        or args.dfa1_vs_hr
        or args.sdnn
        or args.rmssd
        or args.overlay
    )

    if require_features:
        def unmask(x: np.ndarray, mask: np.ndarray):
            """Upsample to the length of the raw HRV signal by inserting NaN's.

            Arguments
            ---------
            x: Downsampled signal matching clean part of HRV signal.
            mask: Mask of clean part of HRV signal.

            Returns
            -------
            Upsampled signal matching length of raw (corrupted) HRV signal.
            """
            y = np.full_like(mask, np.nan, dtype=x.dtype)
            y[mask] = x
            return y

        if args.dfa1_mode == "per_window":
            features = hrv.measures.features_from_sliding_window(hrv_signals)
        elif args.dfa1_mode == "batch":
            features = hrv.measures.features_from_sliding_window_2(hrv_signals)
            # TODO: Apply a similar upsampling on the 'per_window' mode above.
            features = {
                # FIXME: The upsampling restores the length of the raw HRV
                # signal but does not match the sampling of the regular
                # signals.
                # TODO: Match the time samples features["timestamp"] of the HRV
                # signals to the independently recorded signals["timestamp"].
                # Probably need to add up durations of skipped beats in
                # features["timestamp"].
                name: unmask(signal, mask_valid)
                for name, signal in features.items()
            }

    if args.cwt:
        plot.cwt(rr_raw, mask_valid)

    if args.swt:
        plot.swt(rr_raw, mask_valid)

    if args.pointcarre:
        plot.pointcarre(rr_raw, mask_valid=mask_valid)
        plot.pointcarre(rr)

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

    if args.sdnn:
        sdnn = features["sdnn"]
        datetime = features["datetime"]
        fig, ax = plot.series(sdnn, x=datetime)
        fig.autofmt_xdate()
        ax.set_title("sdnn")
        ax.set_ylabel("sdnn")

    if args.rmssd:
        rmssd = features["rmssd"]
        datetime = features["datetime"]
        fig, ax = plot.series(rmssd, x=datetime)
        fig.autofmt_xdate()
        ax.set_title("rmssd")
        ax.set_ylabel("rmssd")

    if args.dfa1:
        # Assuming a constant effort.
        mean_alpha1 = round(np.nanmean(features['alpha1']), 2)
        print(f"mean alpha1 = {mean_alpha1:.2f}")

        plot.df_alpha1(features)

    if args.dfa1_vs_hr:
        plot.df_alpha1_vs_hr(features)

    if args.overlay:
        plot.overlay(signals, features)

    plot.show()


if __name__ == "__main__":
    main()
