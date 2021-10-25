from flask import Flask, render_template

import activity


app = Flask(__name__)


class Activities:
    def __init__(self, db="activities.db"):
        self.db = activity.db_connect(db)

    def list(self):
        cursor = self.db.cursor()
        query = "select * from activities"
        output = cursor.execute(query)
        return output


@app.route("/", methods=["GET"])
def index():
    activities = Activities()
    items = activities.list()
    activity_list_html = "\n".join(
        str(item) + r"<br/>"
        for item in items
    )
    html = f"""<html><body>{activity_list_html}</body></html>"""
    return html


if __name__ == '__main__':
    app.run(port=8080)
