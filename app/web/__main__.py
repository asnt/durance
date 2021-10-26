from flask import Flask, render_template
from sqlalchemy import select

import app.model


flask_app = Flask(__name__)


@flask_app.route("/", methods=["GET"])
def index():
    _ = app.model.make_engine()
    session = app.model.make_session()
    Activity = app.model.Activity
    fields = dict(
        name=Activity.name,
        type=Activity.type,
        duration=Activity.duration,
        distance=Activity.distance,
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
