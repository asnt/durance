import datetime
import importlib
from typing import Any, Dict, Optional

from flask import Flask, render_template, request
import bokeh.embed
import bokeh.models
import bokeh.plotting
import numpy as np
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
def format_generic_to_nothing(data: Optional[Any]) -> Optional[Any]:
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


def make_figure_activities_history(series: Dict) -> bokeh.plotting.Figure:
    figure = bokeh.plotting.figure(height=192,
                                   sizing_mode="stretch_width",
                                   x_axis_type="datetime")

    # Align activity on date, not on precise time.
    dates = [datetime_.replace(hour=0, minute=0, second=0)
             for datetime_ in series["datetime_start"]]
    series["date"] = dates

    # XXX: Assume time on the y axis.
    time_tick_formatter = bokeh.models.NumeralTickFormatter(format="00:00:00")
    interval_minutes = 60
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
    for axis in figure.yaxis:
        axis.major_tick_in = 0
        axis.minor_tick_line_alpha = 0
        axis.axis_line_alpha = 0

    # Only transmit plotted series.
    x = "date"
    y = "duration"
    y_background = "duration_cumulated"
    fields = (x, y, y_background)
    series_shown = {field: series[field] for field in fields}
    source = bokeh.models.ColumnDataSource(series_shown)

    figure.vbar(
        x=x,
        top=y_background,
        source=source,
        width=datetime.timedelta(days=0.8),
        color="lightgray",
    )
    figure.vbar(
        x=x,
        top=y,
        source=source,
        width=datetime.timedelta(days=0.8),
    )

    return figure


def _activity_records_to_arrays(records: list[Dict]) -> Dict[str, np.ndarray]:
    arrays = dict(
        datetime_start=[],
        name=[],
        sport=[],
        sub_sport=[],
        workout=[],
    )
    for field in arrays:
        arrays[field] = [
            getattr(activity, field) for activity in records
        ]
    return arrays


def _summary_records_to_arrays(records: list[Dict]) -> Dict[str, np.ndarray]:
    arrays = dict(
        duration=[],
        distance=[],
        speed=[],
        ascent=[],
        descent=[],
        heart_rate=[],
        step_rate=[],
    )
    for field in arrays:
        arrays[field] = [
            getattr(summary, field) for summary in records
        ]
    return arrays


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

    activities = []
    summaries = []
    script = ""
    div = ""

    if rows:
        activity_records, summary_records = zip(*rows)
        activities = _activity_records_to_arrays(activity_records)
        summaries = _summary_records_to_arrays(summary_records)

        history = activities | summaries

        df = pd.DataFrame(history)
        # Insert a dummy activity for today if none exists, for the resampling
        # and cumulated statistics below to be computed until today.
        last_date = df["datetime_start"].nlargest(1).dt.date.values[0]
        today = datetime.date.today()
        if today > last_date:
            df = df.append(
                {"datetime_start": datetime.datetime.today()},
                ignore_index=True,
            )
        # Resample with a frequency of one day (i.e. add missing days).
        resampler = df.resample("1D", on="datetime_start")
        df_daily = resampler.apply(dict(
            # XXX: Insert daily aggregation for other columns here when needed.
            # Only displaying duration for now.
            duration=np.sum,
        ))
        df_daily = df_daily.reset_index()
        # Compute cumulated duration of past 7 days.
        rolling_week = df_daily.rolling("7D", on="datetime_start")
        df_daily["duration_cumulated"] = rolling_week["duration"].sum()

        history_daily = df_daily.to_dict(orient="list")

        figure = make_figure_activities_history(history_daily)
        script, div = bokeh.embed.components(figure)

    return render_template(
        "activities.html",
        sports=sports,
        sport=sport,
        date_min=date_min,
        date_max=date_max,
        activities=activity_records,
        summaries=summary_records,
        history_div=div,
        history_script=script,
    )


def make_activity_plots(
    series: Dict[str, np.ndarray],
    series_hrv: Dict[str, np.ndarray],
) -> bokeh.model.Model:
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
    # Need both forwards and backwards interpolation to handle missing
    # extremities.
    data = data.interpolate()
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
