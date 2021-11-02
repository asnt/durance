import datetime

from flask import Flask, render_template
from sqlalchemy import select

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
    import numpy as np
    import pandas as pd

    _ = app.model.make_engine()
    session = app.model.make_session()
    Activity = app.model.Activity
    Recording = app.model.Recording
    query = select(Recording.name, Recording.array) \
        .where(Recording.activity_id == Activity.id)
    recordings_data = session.execute(query).all()
    recordings_data = dict(recordings_data)
    data = pd.DataFrame({
        "time": recordings_data["timestamp"],
        "alpha1": recordings_data["heart_rate"],
        "heartrate": recordings_data["heart_rate"],
        "rmssd": recordings_data["heart_rate"],
        "sdnn": recordings_data["heart_rate"],
    })

    import importlib
    plot = importlib.import_module("hrv.plot.bokeh")
    plot_ = plot.overlay(data)
    import bokeh.embed
    script, div = bokeh.embed.components(plot_)
    return render_template(
        "activity.html",
        id_=id_,
        script=script,
        div=div,
    )


if __name__ == '__main__':
    flask_app.run(port=8080)
