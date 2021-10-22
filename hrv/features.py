import numpy as np
import pandas as pd

import hrv.dfa


def compute_features(df):
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
        alpha1 = hrv.dfa.compute_dfa(rr_window_s.copy(), n_scales_max=n_scales_max)

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


def compute_features_2(df):
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
    alpha1 = hrv.dfa.compute_dfa_batch(rr, n_scales_max=n_scales_max)
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
