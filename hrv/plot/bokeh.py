from typing import Optional, Union

import bokeh as bk
import bokeh.layouts
import bokeh.models
import bokeh.plotting
import numpy as np
import pandas as pd


_layout = None


def show():
    if _layout is None:
        raise RuntimeError("no plot defined")
    bk.plotting.show(_layout)


def clear():
    global _layout
    _layout = None


def overlay(df: pd.DataFrame) -> bk.models.LayoutDOM:
    source = bk.models.ColumnDataSource(df)

    plot = bk.plotting.figure()
    plot.grid.visible = False

    x_min = 0
    x_max = len(source)
    dfa_thresholds = [0.5, 0.75, 1.0]
    for threshold in dfa_thresholds:
        plot.line(
            x=[x_min, x_max],
            y=[threshold, threshold],
            # Lighter than light gray.
            line_color="gainsboro",
        )
    plot.y_range = bk.models.Range1d(0, 2)
    dfa_color = "darkslategray"
    plot.yaxis.axis_label = "dfa index"
    plot.yaxis.axis_label_text_color = dfa_color
    plot.line(
        x="index",
        y="alpha1",
        source=source,
        line_color=dfa_color,
        line_width=3,
    )

    config = dict(
        heartrate=dict(
            line=dict(line_color="orange", line_width=2, line_alpha=0.5),
            axis=dict(side="left"),
        ),
        rmssd=dict(
            line=dict(
                line_color="gray",
                line_width=2,
            ),
            axis=dict(side="right"),
        ),
        sdnn=dict(
            line=dict(
                line_color="lightgray",
                line_width=2,
            ),
            axis=dict(side="right"),
        ),
    )
    plot.extra_y_ranges = {}
    for measure, params in config.items():
        plot.line(
            x="index",
            y=measure,
            y_range_name=measure,
            source=source,
            **params["line"],
        )
        if "range_" in params["axis"]:
            range_ = params["axis"]["range_"]
        else:
            y = df[measure].values
            range_ = y.min(), y.max()
        plot.extra_y_ranges[measure] = bk.models.Range1d(*range_)
        axis = bk.models.LinearAxis(y_range_name=measure)
        axis.axis_label = measure
        axis.axis_label_text_color = params["line"]["line_color"]
        plot.add_layout(axis, params["axis"]["side"])

    layout = bk.layouts.row(plot, sizing_mode="stretch_both")

    global _layout
    _layout = layout

    return layout


def recordings_overlay(source: bk.models.ColumnDataSource) \
        -> bk.plotting.Figure:
    """Plot standard recordings (i.e. non-HRV) of an activity."""
    x_measures = ("distance", "time")
    y_measures = list(source.column_names)
    y_measures.remove("index")
    for x_measure in x_measures:
        if x_measure in y_measures:
            y_measures.remove(x_measure)

    figure = bk.plotting.figure()
    figure.grid.visible = False

    config = dict(
        altitude=dict(
            type="line",
            style=dict(color="gray"),
            line=dict(line_color="gray", line_width=2),
            axis=dict(side="left"),
        ),
        heart_rate=dict(
            type="line",
            style=dict(color="orange"),
            line=dict(line_color="orange", line_width=2),
            axis=dict(side="left"),
        ),
        step_rate=dict(
            type="scatter",
            style=dict(color="blue"),
            axis=dict(side="left"),
        ),
    )
    figure.extra_y_ranges = {}
    for measure in y_measures:
        params = config.get(measure, dict(type="line",
                                          style=dict(),
                                          line=dict(),
                                          axis=dict(side="left")))
        if params["type"] == "line":
            figure.line(
                # XXX: Use time or distance on the x axis.
                x="index",
                y=measure,
                y_range_name=measure,
                source=source,
                name=measure,
                **params.get("line", {}),
            )
        elif params["type"] == "scatter":
            figure.scatter(
                # XXX: Use time or distance on the x axis.
                x="index",
                y=measure,
                y_range_name=measure,
                source=source,
                name=measure,
                **params.get("scatter", {}),
            )
        if "range_" in params["axis"]:
            range_ = params["axis"]["range_"]
        else:
            y = source.data[measure]
            range_ = y.min(), y.max()
        figure.extra_y_ranges[measure] = bk.models.Range1d(*range_)
        axis = bk.models.LinearAxis(y_range_name=measure)
        axis.axis_label = measure
        axis.axis_label_text_color = params["style"].get("color", "black")
        figure.add_layout(axis, params["axis"].get("side", "left"))

    return figure


series_config = dict(
    altitude=dict(
        color="gray",
    ),
    heart_rate=dict(
        color="orange",
        y_range=(90, 200),
    ),
    step_rate=dict(
        type_="scatter",
        color="blue",
        y_range=(100, 200),
    ),
    speed=dict(
        y_range=(5, 20),
    ),
)


def series(
    source: bk.models.ColumnDataSource,
    *,
    y: str,
    x: str = "x",
    type_: str = "line",
    color: str = "black",
    line_width: int = 2,
    y_range: Optional[tuple[int, int]] = None,
) -> list[bk.plotting.Figure]:
    """Plot a series for an activity."""
    figure = bk.plotting.figure(height=128)
    figure.grid.visible = False

    if type_ == "line":
        figure.line(
            x=x,
            y=y,
            source=source,
            name=y,
            color=color,
        )
    elif type_ == "scatter":
        figure.scatter(
            x=x,
            y=y,
            source=source,
            name=y,
            color=color,
        )
    y_data = source.data[y]
    y_range = y_range if y_range is not None else (y_data.min(), y_data.max())
    figure.y_range = bk.models.Range1d(*y_range)
    figure.yaxis[0].axis_label = y
    figure.yaxis[0].axis_label_text_color = color

    return figure


histogram_config = dict(
    heart_rate=dict(
        bins_range=(90, 200),
    ),
    speed=dict(
        bins_range=(5, 20),
    ),
    stride_rate=dict(
        bins_range=(100, 200),
    ),
)


def histogram(
    array: np.ndarray,
    bins_range: Optional[tuple[int, int]] = None,
    n_bins: int = 64,
    direction: str = "horizontal",
) -> bk.plotting.Figure:
    """Plot the histogram of an array."""
    assert direction in ("vertical", "horizontal")

    figure = bk.plotting.figure(height=128)
    figure.grid.visible = False
    figure.toolbar_location = None
    figure.tools = []

    bins: Union[int, np.ndarray]
    if bins_range is not None:
        bins = np.linspace(bins_range[0], bins_range[1], n_bins + 1)
    else:
        bins = n_bins
    frequencies, edges = np.histogram(array, density=True, bins=bins)
    if direction == "vertical":
        bottom = 0
        top = frequencies
        left = edges[:-1]
        right = edges[1:]
    else:
        left = 0
        right = frequencies
        bottom = edges[:-1]
        top = edges[1:]
    figure.quad(top=top, bottom=bottom, left=left, right=right)

    if direction == "vertical":
        figure.y_range.start = 0
        if bins_range is not None:
            figure.x_range = bk.models.Range1d(*bins_range)
    else:
        figure.x_range.start = 0
        if bins_range is not None:
            figure.y_range = bk.models.Range1d(*bins_range)

    return figure
