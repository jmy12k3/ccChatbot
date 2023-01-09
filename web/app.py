# coding=utf-8
import os
import sys
import time
import threading

from flask import Flask, render_template, request, jsonify

sys.path.append(os.path.dirname(os.getcwd()))
import data_util
import train
from config import getConfig


gConfig = {}
gConfig = getConfig.get_config()

tok = data_util.tok


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
    req_msg = " ".join(tok(req_msg))
    res_msg = train.predict(req_msg)
    return jsonify({"text": res_msg})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8808)
