import datetime
import importlib

from flask import Flask, render_template
import bokeh.embed
import pandas as pd
import sqlalchemy as sa

import app.model


flask_app = Flask(__name__)


# Expose the zip built-in inside Jinja templates.
flask_app.jinja_env.globals.update(zip=zip)


@flask_app.template_filter("seconds_to_dhms")
def format_seconds_to_days_hours_minutes_seconds(seconds):
    duration = datetime.timedelta(seconds=seconds)
    return str(duration)


@flask_app.template_filter("meters_to_km")
def format_meters_to_km(meters):
    if meters == 0 or meters is None:
        return "-"
    return f"{meters / 1000:.1f}"


@flask_app.route("/", methods=["GET"])
def index():
    _ = app.model.make_engine()
    session = app.model.make_session()
    Activity = app.model.Activity
    fields = {
        "datetime": Activity.datetime_start,

        "name": Activity.name,
        "sport": Activity.sport,
        "sub_sport": Activity.sub_sport,
        "workout": Activity.workout,

        "duration": Activity.duration,
        "distance (km)": Activity.distance,

        "HR (median)": Activity.heartrate_median,
    }
    date_min = None
    date_max = None
    if date_max is None:
        date_max = datetime.date.today()
    if date_min is None:
        date_min = datetime.date.fromtimestamp(0)
    query = sa.select(Activity.id, *fields.values())
    query = query.where(
        sa.and_(
            Activity.datetime_start >= date_min,
            Activity.datetime_start <= date_max,
        )
    )
    query = query.order_by(Activity.datetime_start.desc())
    activity_data = session.execute(query).all()
    activity_ids = [values[0] for values in activity_data]
    activity_values = [values[1:] for values in activity_data]

    pairs = zip(fields, zip(*activity_values))
    calendar_data = dict(pairs)
    calendar_data["active"] = [1] * len(list(calendar_data.values())[0])
    # time = next(zip(*activity_values))
    # calendar_data = {
    #     "time": time,
    #     "active": [1] * len(time),
    #     "duration":
    # }
    import bokeh.models
    data_source = bokeh.models.ColumnDataSource(calendar_data)

    import bokeh.plotting
    figure = bokeh.plotting.figure(height=128, sizing_mode="stretch_width",
                                   x_axis_type="datetime")
    figure.grid.visible = False
    scatter = figure.scatter(
        x="datetime",
        # y="active",
        y="duration",
        # y="HR (median)",
        source=data_source,
    )
    # scatter.xaxis = bokeh.models.DatetimeAxis()
    script, div = bokeh.embed.components(figure)

    return render_template(
        "activities.html",
        activity_ids=activity_ids,
        activity_fields=list(fields.keys()),
        activity_values=activity_values,
        calendar_div=div,
        calendar_script=script,
    )


@flask_app.route("/activity/<id_>", methods=["GET"])
def view_activity(id_):
    _ = app.model.make_engine()
    session = app.model.make_session()

    Activity = app.model.Activity
    fields = {
        "datetime": Activity.datetime_start,

        "name": Activity.name,
        "sport": Activity.sport,
        "sub_sport": Activity.sub_sport,
        "workout": Activity.workout,

        "duration": Activity.duration,
        "distance (km)": Activity.distance,

        "HR (median)": Activity.heartrate_median,
    }
    query = sa.select(*fields.values()) \
        .where(Activity.id == id_)
    activity_values = session.execute(query).one()
    activity_data = dict(zip(fields.keys(), activity_values))

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
        cadence = recordings_data["cadence"]
        # Cadence [steps/foot/minute] to stride rate [strides/minutes].
        recordings_data["stride_rate"] = 2 * cadence
        del recordings_data["cadence"]

    # XXX: For debugging.
    # for k, v in recordings_data.items():
    #     import numpy as np
    #     print(k, v.shape, v.dtype, np.isnan(v).sum())
    x_series_names = ("time", "distance")
    y_series_names = ("altitude", "stride_rate", "heart_rate", "speed")
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
    allowed_y_measures = ("altitude", "stride_rate", "heart_rate", "speed")
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
    if "stride_rate" in data_source.data:
        range_stride_running = (150, 200)
        series_plots["stride_rate_running"] = plot.series(
            data_source,
            y="stride_rate",
            type_="scatter",
            y_range=range_stride_running,
        )
        histograms["stride_rate_running"] = plot.histogram(
            recordings_data["stride_rate"],
            bins_range=range_stride_running,
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
