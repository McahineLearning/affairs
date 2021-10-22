import os
from re import template
import flask
from flask import Flask, render_template, request, jsonify
import yaml
import joblib
import numpy as np


params_path = "params.yaml"
webapp_root = "webappp"

static_dir_path = os.path.join(webapp_root, "static")
template_dir_path = os.path.join(webapp_root, "templates")


app = Flask(__name__, static_folder= static_dir_path, template_folder= template_dir_path)

@app.route("/", methods = ['GET', 'POST'] )
def index():
    if request.method == 'POST':
        pass
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(host= "0.0.0.0", port= 5000, debug= True)