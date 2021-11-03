import bokeh as bk
import bokeh.layouts
import bokeh.models
import bokeh.plotting
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
    x_max = len(df)
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


def recordings(df: pd.DataFrame) -> bk.plotting.Figure:
    """Plot standard recordings of an activity."""
    source = bk.models.ColumnDataSource(df)

    x_measures = ("distance", "time")
    y_measures = list(df.columns)
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
        cadence=dict(
            type="scatter",
            style=dict(color="blue"),
            axis=dict(side="left"),
        ),
        stride_rate=dict(
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
                **params.get("line", {}),
            )
        elif params["type"] == "scatter":
            figure.scatter(
                # XXX: Use time or distance on the x axis.
                x="index",
                y=measure,
                y_range_name=measure,
                source=source,
                **params.get("scatter", {}),
            )
        if "range_" in params["axis"]:
            range_ = params["axis"]["range_"]
        else:
            y = df[measure].values
            range_ = y.min(), y.max()
        figure.extra_y_ranges[measure] = bk.models.Range1d(*range_)
        axis = bk.models.LinearAxis(y_range_name=measure)
        axis.axis_label = measure
        axis.axis_label_text_color = params["style"].get("color", "black")
        figure.add_layout(axis, params["axis"].get("side", "left"))

    return figure
