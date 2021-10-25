from flask import Flask, render_template

import activity

# from bokeh.client import pull_session
# from bokeh.embed import server_session

app = Flask(__name__)

# @app.route('/', methods=['GET'])
# def bkapp_page():

#     # pull a new session from a running Bokeh server
#     with pull_session(url="http://localhost:5006/sliders") as session:

#         # update or customize that session
#         session.document.roots[0].children[1].title.text = "Special sliders for a specific user!"

#         # generate a script to load the customized session
#         script = server_session(session_id=session.id, url='http://localhost:5006/sliders')

#         # use the script in the rendered page
#         return render_template("embed.html", script=script, template="Flask")


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
