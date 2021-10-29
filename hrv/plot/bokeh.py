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
