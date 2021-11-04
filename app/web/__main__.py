import datetime
import importlib

from flask import Flask, render_template
from sqlalchemy import select
import bokeh.embed
import pandas as pd

import app.model


flask_app = Flask(__name__)


# Make the zip built-in usable inside Jinja templates.
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
    fields = dict(
        datetime_start=Activity.datetime_start,

        name=Activity.name,
        sport=Activity.sport,
        sub_sport=Activity.sub_sport,
        workout=Activity.workout,

        duration=Activity.duration,
        distance=Activity.distance,

        heartrate_mean=Activity.heartrate_mean,
        heartrate_median=Activity.heartrate_median,
    )
    query = select(Activity.id, *fields.values())
    activity_data = session.execute(query).all()
    activity_ids = [values[0] for values in activity_data]
    activity_values = [values[1:] for values in activity_data]
    return render_template(
        "activities.html",
        activity_ids=activity_ids,
        activity_fields=list(fields.keys()),
        activity_values=activity_values,
    )


@flask_app.route("/activity/<id_>", methods=["GET"])
def view_activity(id_):
    _ = app.model.make_engine()
    session = app.model.make_session()
    Recording = app.model.Recording
    query = select(Recording.name, Recording.array) \
        .where(Recording.activity_id == id_)
    recordings_data = session.execute(query).all()

    recordings_data = dict(recordings_data)
    # XXX: For debugging.
    # for k, v in recordings_data.items():
    #     print(k, v.shape, v.dtype)
    series_names = ("altitude", "cadence", "heart_rate", "speed")
    recordings_series = {
        name: series
        for name, series in recordings_data.items()
        if name in series_names
    }
    if "speed" in recordings_series:
        # From [metres/second] to [kilometres/hour].
        recordings_series["speed"] *= 1e-3 * 3600
    if "cadence" in recordings_series:
        cadence = recordings_series["cadence"]
        # Cadence [steps/foot/minute]
        # to
        # stride rate [steps/minutes] (steps from both feet).
        recordings_series["stride_rate"] = 2 * cadence
        del recordings_series["cadence"]
    data = pd.DataFrame.from_dict(recordings_series)

    # TODO: Add map plot.
    # positions_names = ("position_lat", "position_long")

    plot = importlib.import_module("hrv.plot.bokeh")
    data_source = bokeh.models.ColumnDataSource(data)
    figure = plot.recordings(data_source)

    series_choice = bokeh.models.CheckboxButtonGroup(
        labels=data_source.column_names,
        active=[0],
    )
    series_choice_clicked = bokeh.models.CustomJS(
        args=dict(source=data_source),
        # FIXME: This does not seem to work because the plot on the client side
        # expects all series to be present, as defined on the server.
        # TODO: Find a way to hide some series?
        code="""
    console.log(source.data);
    const data = source.data;
    const series = cb_obj.value
    source.data = Object.fromEntries(
        Object.entries(data).filter(([key]) => key === "heart_rate")
    )
    source.change.emit();
    console.log(source.data);
""",
    )
    series_choice.js_on_click(series_choice_clicked)

    column = bokeh.layouts.column
    row = bokeh.layouts.row
    layout_series_choice = row(series_choice)
    layout_figure = row(figure, sizing_mode="stretch_width")
    layout = column([layout_series_choice, layout_figure],
                    sizing_mode="stretch_width")

    script, div = bokeh.embed.components(layout)

    return render_template(
        "activity.html",
        id_=id_,
        script=script,
        div=div,
    )


if __name__ == '__main__':
    flask_app.run(port=8080)
