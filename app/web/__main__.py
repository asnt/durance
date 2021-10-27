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
    if meters == 0:
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
    query = select(*fields.values())
    activities = session.execute(query).all()
    return render_template(
        "activities.html",
        activity_fields=list(fields.keys()),
        activities=activities,
    )


@flask_app.route("/activity/<id_>", methods=["GET"])
def view_activity(id_):
    return id_


if __name__ == '__main__':
    flask_app.run(port=8080)
