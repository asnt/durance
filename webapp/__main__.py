from flask import Flask, render_template

import activity


app = Flask(__name__)


class Activities:
    def __init__(self, db="activities.db"):
        self.db = activity.db_connect(db)

    def values(self):
        cursor = self.db.cursor()
        query = "select * from activities"
        output = cursor.execute(query)
        return output


@app.route("/", methods=["GET"])
def index():
    activities = Activities()
    activities_values = activities.values()
    return render_template("activities.html",
                           activities_values=activities_values)


@app.route("/activity/<id_>", methods=["GET"])
def view_activity(id_):
    return id_
    # activities = Activities()
    # activities_values = activities.values()
    # return render_template("activity.html", activity_values=activity_values)


if __name__ == '__main__':
    app.run(port=8080)
