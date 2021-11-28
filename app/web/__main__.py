import datetime
import importlib
from typing import Any, Dict, Optional

from flask import Flask, render_template, request
import bokeh.embed
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
def format_meters_to_km(meters):
    if meters == 0 or meters is None:
        return "-"
    return f"{meters / 1000:.1f}"


def make_figure_activities_history(series: Dict) -> bokeh.plotting.Figure:
    import bokeh.plotting
    figure = bokeh.plotting.figure(height=128, sizing_mode="stretch_width",
                                   x_axis_type="datetime")

    dates = [datetime_.date() for datetime_ in series["datetime_start"]]

    import collections
    dates_per_month = collections.defaultdict(list)
    for date in dates:
        month = date.month
        dates_per_month[month].append(date)

    def min_date(dates):
        import functools
        return functools.reduce(
            lambda x, y: min(x, y),
            dates,
            datetime.date.max,
        )

    min_date_per_month = {
        month: min_date(dates)
        for month, dates in dates_per_month.items()
    }

    axis_months = bokeh.models.DatetimeAxis()
    # XXX: Not sure how to define this. Does not work for less than "11".
    months_invervals = list(range(11))
    axis_months.ticker = bokeh.models.MonthsTicker(months=months_invervals)
    axis_months.formatter.days = ["%m"]
    axis_months.formatter.months = ["%m"]
    axis_months.formatter.years = ["%m"]
    figure.add_layout(axis_months, "below")

    dates_per_year = collections.defaultdict(list)
    for date in dates:
        year = date.year
        dates_per_year[year].append(date)
    min_date_per_year = {
        year: min_date(dates)
        for year, dates in dates_per_year.items()
    }
    # TODO: Place a tick on the first day of each yet in the visible range,
    # and on the first days of the visible range.
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

    return figure.vbar(
        x=x,
        top=y,
        source=source,
        # FIXME: Does not work as indicated in the doc. Bug?
        # https://docs.bokeh.org/en/latest/docs/reference/plotting/figure.html#bokeh.plotting.Figure.vbar
        # width=10,
        line_width=2,
    )


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
        .where(sa.and_(
            Activity.datetime_start >= date_min,
            Activity.datetime_start < date_max_plus_1_day
        ))
        .order_by(Activity.datetime_start.desc())
        .join(Summary)
    )

    _ = app.model.make_engine()
    session = app.model.make_session()
    rows = session.execute(query).all()

    activities, summaries = zip(*rows)

    activity_fields = (
        "datetime_start",
        "name",
        "sport",
        "sub_sport",
        "workout",
    )
    activity_series = {
        field: [getattr(activity, field) for activity in activities]
        for field in activity_fields
    }

    summary_fields = (
        "duration",
        "distance",
        "speed",
        "ascents",
        "descents",
        "heart_rate",
        "step_rate",
    )
    summary_series = {
        field: [getattr(summary, field) for summary in summaries]
        for field in summary_fields
    }

    data_series = activity_series | summary_series

    figure = make_figure_activities_history(data_series)
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


@flask_app.route("/activity/<id_>", methods=["GET"])
def view_activity(id_):
    _ = app.model.make_engine()
    session = app.model.make_session()

    Activity = app.model.Activity
    activity_fields = {
        "datetime": Activity.datetime_start,

        "name": Activity.name,
        "sport": Activity.sport,
        "sub_sport": Activity.sub_sport,
        "workout": Activity.workout,

        # "duration": Activity.duration,
        # "distance (km)": Activity.distance,

        # "HR (median)": Activity.heartrate_median,
    }
    query = sa.select(*activity_fields.values()) \
        .where(Activity.id == id_)
    activity_values = session.execute(query).one()
    activity_data = dict(zip(activity_fields.keys(), activity_values))

    Recording = app.model.Recording
    query = sa.select(Recording.name, Recording.array) \
        .where(Recording.activity_id == id_)
    recordings_data = session.execute(query).all()

    recordings_data = dict(recordings_data)

    # TODO: Add the relative time from the start of the activity.
    recordings_data["time"] = recordings_data["timestamp"]
    del recordings_data["timestamp"]

    if "speed" in recordings_data:
        # From m/s to km/h.
        recordings_data["speed"] *= 1e-3 * 3600
    if "cadence" in recordings_data:
        # strides/minute
        stride_rate = recordings_data.pop("cadence")
        # steps/minute
        recordings_data["step_rate"] = 2 * stride_rate

    # XXX: For debugging.
    # for k, v in recordings_data.items():
    #     import numpy as np
    #     print(k, v.shape, v.dtype, np.isnan(v).sum())
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

    plot = importlib.import_module("hrv.plot.bokeh")
    data_source = bokeh.models.ColumnDataSource(data)

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
        name: plot.histogram(data[name].values,
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
            recordings_data["step_rate"],
            bins_range=range_step_running,
        )

    if "rr" in recordings_data:
        hrv_source = bokeh.models.ColumnDataSource()
        hrv_source.add(recordings_data["rr"], "rr")
        import numpy as np
        hrv_source.add(np.arange(len(recordings_data["rr"])), "x")
        series_plots["rr"] = plot.series(hrv_source,
                                         y="rr",
                                         type_="scatter")
        histograms["rr"] = plot.histogram(recordings_data["rr"])

        relevant_range = (0.3, 0.65)
        series_plots["rr_relevant"] = plot.series(
            hrv_source,
            y="rr",
            type_="scatter",
            y_range=relevant_range,
        )
        histograms["rr_relevant"] = plot.histogram(
            recordings_data["rr"],
            bins_range=relevant_range,
        )

    # figure = plot.recordings_overlay(data_source)
    #
    # series_names = data_source.column_names
    # series_names.remove("index")
    # default_active_names = ["heart_rate"]
    # active = [
    #     series_names.index(name)
    #     for name in default_active_names
    #     if name in series_names
    # ]
    # series_choice = bokeh.models.CheckboxButtonGroup(
    #     labels=series_names,
    #     active=active,
    # )
    # series_choice_clicked = bokeh.models.CustomJS(
    #     args=dict(figure=figure),
    #     code="""
    # const labels = this.labels;
    # const active = this.active;
    # let visible = labels.map(_ => false);
    # for (let index of active) {
    #     visible[index] = true;
    # }
    # for (let index in labels) {
    #     const results = figure.select(labels[index]);
    #     if (results.length == 0) {
    #         console.error("plot no found");
    #         continue;
    #     }
    #     const plot = results[0];
    #     plot.visible = visible[index];
    # }
# """,
    # )
    # series_choice.js_on_click(series_choice_clicked)

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

    script, div = bokeh.embed.components(layout)

    return render_template(
        "activity.html",
        id_=id_,
        activity=activity_data,
        script=script,
        div=div,
    )


if __name__ == '__main__':
    flask_app.run(port=8080)
