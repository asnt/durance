# -*- coding: utf-8 -*-
"""Compute FDA-alpha1 index from reart rate variability (HRV).

This can be used for estimating the aerobic threshold.
"""
import argparse
import csv
import math

import fitparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine
import seaborn as sn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fit")
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


def plot_df_rr(df):
    pn = plotnine
    (
        pn.ggplot(df)
        + pn.aes(x="time", y="rr")
        + pn.geom_line()
        + pn.labs(title="RR intervals", x="seconds", y="milliseconds")
    )


def compute_dfa(pp_values, lower_scale_limit, upper_scale_limit):
    # Scales DFA is conducted between lower_scale_limit and upper_scale_limit.
    scale_density = 30

    # Order of polynomial fit.
    order = 1

    # Initialize. Using logarithmic scales.
    start = np.log(lower_scale_limit) / np.log(10)
    stop = np.log(upper_scale_limit) / np.log(10)
    scales = np.floor(
        np.logspace(np.log10(math.pow(10, start)),
                    np.log10(math.pow(10, stop)),
                    scale_density)
    )
    F = np.zeros(len(scales))
    count = 0

    # Step 1: Determine the "profile" (integrated signal with subtracted
    # offset).
    x = pp_values
    y_n = np.cumsum(x - np.mean(x))

    for scale in scales:
        rms = []
        # Step 2: Divide the profile into N non-overlapping segments of equal
        # length s.
        shape = (int(scale), int(np.floor(len(x) / scale)))
        size = int(shape[0]) * int(shape[1])

        # Beginning to end, here we reshape so that we have a number of
        # segments based on the scale used at this cycle.
        Y_n1 = y_n[0:size].reshape(shape[::-1])
        # End to beginning.
        Y_n2 = (y_n[len(y_n) - size:len(y_n)]).reshape(shape[::-1])

        # Concatenate.
        Y_n = np.vstack((Y_n1, Y_n2))

        # Step 3: Calculate the local trend for each 2Ns segments by a least
        # squares fit of the series.
        for cut in np.arange(0, 2 * shape[1]):
            xcut = np.arange(0, shape[0])
            pl = np.polyfit(xcut, Y_n[cut, :], order)
            Yfit = np.polyval(pl, xcut)
            arr = Yfit - Y_n[cut, :]
            rms.append(np.sqrt(np.mean(arr * arr)))

        if (len(rms) > 0):
            F[count] = np.power(
                (1 / (shape[1] * 2)) * np.sum(np.power(rms, 2)),
                1/2
            )
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


def plot_valid(rr, mask_valid):
    cmap = "hsv"

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(rr)), rr, c=rr, cmap=cmap)

    fig, ax = plt.subplots()
    ax.scatter(rr[:-1], rr[1:], c=rr[:-1], cmap=cmap)

    fig, ax = plt.subplots()
    rr_valid = np.copy(rr)
    rr_valid[~mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_valid)
    rr_invalid = np.copy(rr)
    rr_invalid[mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_invalid)

    plt.show()


def main():
    args = parse_args()

    rr_raw = load_rr_from_fit(args.fit)
    mask_valid = compute_valid_mask(rr_raw)
    rr = rr_raw[mask_valid]
    time_ = np.cumsum(rr)

    print(rr_raw.shape)
    print(time_.shape)
    plot_valid(rr_raw, mask_valid)

    df = pd.DataFrame()
    df["time"] = time_
    df["rr"] = rr

    plot_df_rr(df)

    df_features = compute_features(df)
    print(df_features.head())

    # Assuming a constant effort.
    mean_alpha1 = round(np.mean(df_features['alpha1']), 2)
    print(f"mean alpha1 = {mean_alpha1:.2f}")

    plot_df_alpha1(df_features)

    # Filter further based on SDNN to remove the moments standing still.

    # Based on visual inspection of the data.
    threshold_sdnn = 10

    mask_motion = df_features['sdnn'] < threshold_sdnn
    df_features_motion = df_features.loc[mask_motion]

    mean_alpha1_motion = round(np.mean(df_features_motion['alpha1']), 2)
    print(f"mean alpha1 in motion = {mean_alpha1_motion:.2f}")

    plot_df_alpha1(df_features_motion)


def plot_df_alpha1(df):
    """# Analyzing results

    Let's look at *alpha 1*. According to the papers, you should see the following:


    *   Values close to 1 for very low intensity efforts (40% of VO2max)
    *   Values close 0.75 for the aerobic threshold
    *   Values close to 0.5 for anything beyond the anaerobic threshold
    """
    pn = plotnine
    plot = (
        pn.ggplot(df)
        + pn.aes(x='time', y='alpha1')
        + pn.geom_point()
        + pn.geom_line()
        + pn.scale_y_continuous(
            # This seems to be the range of meaningful values.
            limits=[0, 1.5],
        )
        + pn.labs(
            title=("Plot of alpha 1 as derived from DFA for aerobic threshold"
                   " estimation"),
            x='Window',
            y="alpha 1"
        )
    )
    fig = plot.draw()
    del fig
    plt.show()


if __name__ == "__main__":
    main()
