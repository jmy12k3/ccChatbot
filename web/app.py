# coding=utf-8
import os
import sys
import threading
import time

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import getConfig  # noqa: E402
from predict import translate  # noqa: E402

# region Config

gConfig = {}
gConfig = getConfig.get_config()

UNK = gConfig["unk"]

# endregion


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

    if UNK in res_msg:
        res_msg = "親，我不懂你在說什麼，請再說一次🙏🏻"

    return jsonify({"text": res_msg})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8808)
