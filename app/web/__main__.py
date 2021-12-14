import datetime
import importlib
from typing import Any, Dict, Optional

from flask import Flask, render_template, request
import bokeh.embed
import bokeh.model
import bokeh.plotting
import pandas as pd
import sqlalchemy as sa

import app.model


flask_app = Flask(__name__)


# Expose the zip built-in inside Jinja templates.
flask_app.jinja_env.globals.update(zip=zip)


@flask_app.template_filter("ignore_none")
def format_none_to_nothing(data: Optional[Any]) -> str:
    if data is None:
        return ""
    return data


@flask_app.template_filter("ignore_generic")
def format_genertic_to_nothing(data: Optional[Any]) -> Optional[Any]:
    if data == "generic":
        return ""
    return data


@flask_app.template_filter("seconds_to_dhms")
def format_seconds_to_days_hours_minutes_seconds(
    seconds: Optional[int],
) -> Optional[str]:
    if seconds is None:
        return None
    duration = datetime.timedelta(seconds=seconds)
    return str(duration)


@flask_app.template_filter("meters_to_km")
def format_meters_to_km(meters: int) -> str:
    if meters == 0 or meters is None:
        return ""
    return f"{meters / 1000:.1f}"


def _make_axis_months() -> bokeh.models.DatetimeAxis:
    axis_months = bokeh.models.DatetimeAxis()
    # XXX: Not sure how to define this. Does not work for less than "12".
    months_invervals = list(range(12))
    axis_months.ticker = bokeh.models.MonthsTicker(months=months_invervals)
    axis_months.formatter.days = ["%m"]
    axis_months.formatter.months = ["%m"]
    axis_months.formatter.years = ["%m"]
    return axis_months


def _make_axis_years() -> bokeh.models.DatetimeAxis:
    # TODO: Goal: Place a tick on the first day of each year in the visible
    # range,and on the first days of the visible range.
    #
    # Alternative 1. (Does not work.)
    #
    # axis_years = bokeh.models.Axis()
    # axis_years.formatter = bokeh.models.()
    # FIXME: Not sure how to convert a `date` or `datetime` to a float value
    # that the x axis understands.
    # date_nums = [
    #     datetime.datetime(date.year, date.month, date.day).timestamp()
    #     for date in min_date_per_year.values()
    # ]
    # print(date_nums)
    # axis_years.ticker = bokeh.models.FixedTicker(
    #     ticks=date_nums,
    # )
    #
    # Alternative 2. (Partial solution. Temporary.)
    #
    axis_years = bokeh.models.DatetimeAxis()
    # FIXME: I think this puts a tick at each first day of each year only. Not
    # sure how to adapt place at a late day if the x-axis range does not
    # contain the first day.
    # XXX: This might be sufficient for now.
    axis_years.ticker = bokeh.models.YearsTicker()
    # XXX: Repeat the year on the first day of each month for now.
    # axis_years.ticker = bokeh.models.MonthsTicker(
    axis_years.formatter.days = ["%Y"]
    axis_years.formatter.months = ["%Y"]
    axis_years.formatter.years = ["%Y"]
    return axis_years


def make_figure_activities_history(series: Dict) -> bokeh.plotting.Figure:
    import bokeh.plotting
    figure = bokeh.plotting.figure(height=128,
                                   sizing_mode="stretch_width",
                                   x_axis_type="datetime")

    dates = [datetime_.date() for datetime_ in series["datetime_start"]]

    import collections
    dates_per_month = collections.defaultdict(list)
    for date in dates:
        month = date.month
        dates_per_month[month].append(date)

    axis_months = _make_axis_months()
    figure.add_layout(axis_months, "below")

    axis_years = _make_axis_years()
    figure.add_layout(axis_years, "below")

    figure.xaxis[0].formatter.days = ["%d"]
    # Note: Cover all "days intervals" for each month.
    # `list(range(5))` would display tick only for days 1, 2, 3 and 4.
    days_intervals = list(range(32))
    figure.xaxis[0].ticker = bokeh.models.DaysTicker(days=days_intervals)
    # figure.grid.visible = False

    # XXX: Assume time on the y axis.
    time_tick_formatter = bokeh.models.NumeralTickFormatter(format="00:00:00")
    interval_minutes = 30
    interval_seconds = interval_minutes * 60
    time_ticker = bokeh.models.SingleIntervalTicker(interval=interval_seconds)

    figure.yaxis[0].formatter = time_tick_formatter
    figure.yaxis[0].ticker = time_ticker
    # Duplicate y axis on the right for legibility when the plot is wide.
    yaxis_right = bokeh.models.ContinuousAxis(
        ticker=time_ticker,
        formatter=time_tick_formatter,
    )
    figure.add_layout(yaxis_right, "right")

    figure.xaxis[0].axis_line_alpha = 0
    figure.xaxis[0].major_tick_in = 0
    axis_months.axis_line_alpha = 0
    axis_months.major_tick_in = 0
    axis_years.axis_line_alpha = 0
    axis_years.major_tick_in = 0
    for axis in figure.yaxis:
        axis.major_tick_in = 0
        axis.minor_tick_line_alpha = 0
        axis.axis_line_alpha = 0

    # Only transmit plotted series.
    x = "datetime_start"
    y = "duration"
    series_shown = {field: series[field] for field in (x, y)}
    import bokeh.models
    source = bokeh.models.ColumnDataSource(series_shown)

    figure.vbar(
        x=x,
        top=y,
        source=source,
        # FIXME: Does not work as indicated in the doc. Bug?
        # https://docs.bokeh.org/en/latest/docs/reference/plotting/figure.html#bokeh.plotting.Figure.vbar
        # width=10,
        line_width=3,
    )

    return figure


@flask_app.route("/", methods=["GET"])
def index():
    args = request.args

    sports = ("", "running", "cycling", "swimming", "generic")
    sport = args.get("sport", "")
    if sport not in sports:
        sport = ""

    date_max_str = args.get("date_max", None)
    if date_max_str is None:
        date_max = datetime.date.today()
    else:
        date_max = datetime.date.fromisoformat(date_max_str)
    date_max_plus_1_day = date_max + datetime.timedelta(days=1)

    date_min_str = args.get("date_min", None)
    if date_min_str is None:
        date_min = date_max - datetime.timedelta(weeks=4)
    else:
        date_min = datetime.date.fromisoformat(date_min_str)

    Activity = app.model.Activity
    Summary = app.model.Summary
    query = sa.select(Activity, Summary)
    if sport:
        query = query.where(Activity.sport == sport)
    query = (
        query
        .where(Activity.datetime_start >= date_min)
        .where(Activity.datetime_start < date_max_plus_1_day)
        .order_by(Activity.datetime_start.desc())
        .outerjoin(Summary)
    )

    _ = app.model.make_engine()
    session = app.model.make_session()
    rows = session.execute(query).all()

    activity_history = dict(
        datetime_start=[],
        name=[],
        sport=[],
        sub_sport=[],
        workout=[],
    )

    summary_history = dict(
        duration=[],
        distance=[],
        speed=[],
        ascent=[],
        descent=[],
        heart_rate=[],
        step_rate=[],
    )

    activities = []
    summaries = []
    script = ""
    div = ""

    if rows:
        activities, summaries = zip(*rows)
        for field in activity_history:
            activity_history[field] = [
                getattr(activity, field) for activity in activities
            ]
        for field in summary_history:
            summary_history[field] = [
                getattr(summary, field) for summary in summaries
            ]

        data_history = activity_history | summary_history

        figure = make_figure_activities_history(data_history)
        script, div = bokeh.embed.components(figure)

    return render_template(
        "activities.html",
        sports=sports,
        sport=sport,
        date_min=date_min,
        date_max=date_max,
        activities=activities,
        summaries=summaries,
        history_div=div,
        history_script=script,
    )


def make_activity_plots(series: Dict,
                        series_hrv: Dict
                        ) -> bokeh.model.model.Model:
    plot = importlib.import_module("hrv.plot.bokeh")
    data_source = bokeh.models.ColumnDataSource(series)

    x_measures = ("distance", "index", "time")
    allowed_y_measures = ("altitude", "step_rate", "heart_rate", "speed")
    y_measures = [
        measure
        for measure in allowed_y_measures
        if measure in data_source.data
    ]

    series_plots = {
        name: plot.series(data_source,
                          y=name,
                          **plot.series_config.get(name, {}))
        for name in y_measures
    }

    histograms = {
        name: plot.histogram(series[name].values,
                             **plot.histogram_config.get(name, {}))
        for name in y_measures
    }

    # Plot again, focussing the range on the running (i.e. higher frequency).
    if "step_rate" in data_source.data:
        range_step_running = (160, 190)
        series_plots["step_rate_running"] = plot.series(
            data_source,
            y="step_rate",
            type_="scatter",
            y_range=range_step_running,
        )
        histograms["step_rate_running"] = plot.histogram(
            # recordings_data["step_rate"],
            series["step_rate"],
            bins_range=range_step_running,
        )

    if "rr" in series_hrv:
        rr = series_hrv["rr"]
        hrv_source = bokeh.models.ColumnDataSource()
        hrv_source.add(rr, "rr")
        import numpy as np
        hrv_source.add(np.arange(len(rr)), "x")
        series_plots["rr"] = plot.series(hrv_source, y="rr", type_="scatter")
        histograms["rr"] = plot.histogram(rr)

        relevant_range = (0.3, 0.65)
        series_plots["rr_relevant"] = plot.series(
            hrv_source,
            y="rr",
            type_="scatter",
            y_range=relevant_range,
        )
        histograms["rr_relevant"] = plot.histogram(
            rr,
            bins_range=relevant_range,
        )

    gridplot = bokeh.layouts.gridplot
    layout = gridplot(
        [
            [series_plots[name], histograms[name]]
            for name in sorted(series_plots.keys())
        ],
        sizing_mode="stretch_width",
    )
    for histogram in histograms.values():
        histogram.sizing_mode = "fixed"
        histogram.width = 128

    return layout


@flask_app.route("/activity/<id_>", methods=["GET"])
def view_activity(id_):
    _ = app.model.make_engine()
    session = app.model.make_session()

    Activity = app.model.Activity
    Summary = app.model.Summary
    query = (
        sa
        .select(Activity, Summary)
        .where(Activity.id == id_)
        .outerjoin(Summary)
    )
    activity_summary = session.execute(query).one()
    activity, summary = activity_summary

    Recording = app.model.Recording
    query = sa.select(Recording.name, Recording.array) \
        .where(Recording.activity_id == id_)
    recordings_data = session.execute(query).all()

    recordings_data = dict(recordings_data)

    # TODO: Add the relative time from the start of the activity.
    recordings_data["time"] = recordings_data.pop("timestamp")

    if "speed" in recordings_data:
        # From m/s to km/h.
        recordings_data["speed"] *= 1e-3 * 3600
    if "cadence" in recordings_data:
        # strides/minute
        stride_rate = recordings_data.pop("cadence")
        # steps/minute
        recordings_data["step_rate"] = 2 * stride_rate

    x_series_names = ("time", "distance")
    y_series_names = ("altitude", "step_rate", "heart_rate", "speed")
    recordings_series = {
        name: series
        for name, series in recordings_data.items()
        if name in x_series_names + y_series_names
    }
    data = pd.DataFrame.from_dict(recordings_series)

    # Replace NaN values using neighbors.
    # Forwards.
    data = data.interpolate()
    # Backwards.
    data = data.interpolate(method="backfill")

    # TODO: Add map plot.
    # positions_names = ("position_lat", "position_long")

    # TODO: Allow to choose the x-axis from the browser.
    data["x"] = data.index.values
    # data["x"] = data["time"]
    # data["x"] = data["distance"]

    series = data.to_dict(orient="series")
    series_hrv = dict()
    if "rr" in recordings_data:
        series_hrv["rr"] = recordings_data["rr"]

    model = make_activity_plots(series, series_hrv)
    plots_script, plots_div = bokeh.embed.components(model)

    return render_template(
        "activity.html",
        id_=id_,
        activity=activity,
        summary=summary,
        plots_script=plots_script,
        plots_div=plots_div,
    )


if __name__ == '__main__':
    flask_app.run(port=8080)
