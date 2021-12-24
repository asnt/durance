import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def show():
    plt.show()


def rr(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(rr)), rr, c=rr, cmap=cmap)


def rr_cumsum(rr, mask_valid, cmap="hsv"):
    rr_cumsum = np.cumsum(rr - np.mean(rr))
    fig, ax = plt.subplots()

    rr_valid = rr[mask_valid]
    rr_valid_cumsum = np.cumsum(rr_valid - np.mean(rr_valid))
    rr_valid_cumsum_masked = np.full_like(rr, np.nan)
    rr_valid_cumsum_masked[mask_valid] = rr_valid_cumsum

    x = np.arange(len(rr_cumsum))
    ax.plot(x, rr_cumsum, color="black", alpha=0.25, linewidth=2)
    ax.plot(x, rr_valid_cumsum_masked, color="black")


def pointcarre(rr, mask_valid=None, cmap="hsv"):
    fig, ax = plt.subplots()
    if mask_valid is not None:
        colors = np.full((len(rr) - 1, 3), (.8, .3, .3), dtype=float)
        mask_valid_ = mask_valid[:-1] & mask_valid[1:]
        colors[mask_valid_] = (0, 0, 0)
    else:
        colors = None
    ax.scatter(rr[:-1], rr[1:], c=colors)


def scatter(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    rr_valid = np.copy(rr)
    rr_valid[~mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_valid)
    rr_invalid = np.copy(rr)
    rr_invalid[mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_invalid)


def lines(rr, mask_valid, cmap="hsv"):
    fig, ax = plt.subplots()
    rr_valid = np.copy(rr)
    rr_valid[~mask_valid] = np.nan
    ax.plot(np.arange(len(rr)), rr_valid, color="black")
    rr_invalid = np.copy(rr)
    rr_invalid[mask_valid] = np.nan
    ax.scatter(np.arange(len(rr)), rr_invalid, color="red", alpha=0.25)


def series(y, mask_valid=True, x=None, cmap="hsv"):
    fig, ax = plt.subplots()
    y_valid = np.copy(y)
    y_valid[~mask_valid] = np.nan
    if x is None:
        x = np.arange(len(y))
    ax.plot(x, y, color="black")
    return fig, ax


def cwt(rr, mask_valid, cmap="hsv"):
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


def swt(rr, mask_valid, cmap="hsv"):
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


def df_alpha1(df, cmap="Spectral"):
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


def overlay(df, recordings=None, cmap="Spectral"):
    dfa_thresholds = [0.5, 0.75, 1.0]
    color_normalizer = mpl.colors.Normalize(vmin=dfa_thresholds[0],
                                            vmax=dfa_thresholds[-1])
    dfa_ticks = dfa_thresholds + [1.5, 2.0]

    time = df["time"]
    alpha1 = df["alpha1"].values

    fig, ax = plt.subplots()

    #
    # Raw recordings.
    #

    if recordings is not None:
        # Altitude.
        ax_ = ax.twinx()
        # ax_.spines.left.set_position(("axes", 0 - 1 / 10))
        y = recordings["altitude"]
        x = recordings["timestamp"] * 1e9
        plot_, = ax_.plot(x, y, color="lightgray")
        ax_.fill_between(x, y.min(), y, color="lightgray")
        # ax_.set_ylabel("altitude")
        # ax_.yaxis.label.set_color(plot_.get_color())
        # ax_.tick_params(axis="y", colors=plot_.get_color())

    #
    # Processed signals derived from HRV data.
    #

    time = df["time"].values

    alpha1 = df["alpha1"].values
    color_alpha1 = "dimgray"

    # Create separated axes in order to draw over the axes of the recordings.
    ax_ = ax.twinx()
    plot_dfa1, = ax_.plot(time, alpha1, color=color_alpha1, zorder=2.5)
    ax_.scatter(time, alpha1, c=alpha1, norm=color_normalizer, cmap=cmap)

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
            linewidth=2,
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


def df_alpha1_vs_hr(df, cmap="Spectral"):
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
