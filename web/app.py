# coding=utf-8
import os
import threading
import time

from flask import Flask, jsonify, render_template, request

os.path.append(os.path.dirname(os.path.abspath(__file__)))

from translate import translate  # noqa: E402


def heartbeat():
    print(time.strftime("%Y-%m-%d %H:%M:%S - heartbeat", time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()


timer = threading.Timer(60, heartbeat)
timer.start()

app = Flask(__name__, static_url_path="/static")


@app.route("/message", methods=["POST"])
def reply():
    req_msg = request.form["msg"]
    res_msg = translate(req_msg)
    return jsonify({"text": res_msg})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8808)
