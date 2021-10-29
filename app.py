import re
import flask
import json
from flask import config
import numpy as np
import joblib
import pickle
import os
from flask import Flask, render_template, request
import yaml

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

app = Flask(__name__,template_folder=template_dir)


@app.route("/")
@app.route("/index")

def index():
	return flask.render_template('index.html')


@app.route("/predict",methods = ['POST'])
def make_predictions():

    config = read_params(config_path=params_path)
    model_dir = config['webapp_model_dir']

    with open(model_dir, 'rb') as f:
        reg = joblib.load(f)

    if request.method == 'POST':
        
        rateMarriage = float(request.form['rateMarriage'])
        age = float(request.form['age'])
        yearsMarried = float(request.form['yearsMarried'])
        children = float(request.form['children'])
        religious = float(request.form['religious'])
        educ = float(request.form['educ'])
        occupation = float(request.form['occupation'])
        husbandOccupation = float(request.form['husbandOccupation'])

        

        occupation_self_1 = 0
        occupation_self_2 = 0
        occupation_self_3 = 0
        occupation_self_4 = 0
        occupation_self_5 = 0
        occupation_self_6 = 0
        if occupation == 1:
            occupation_self_1 = 0
        elif occupation == 2:
            occupation_self_2 = 1
        elif occupation == 3:
            occupation_self_3 = 1
        elif occupation == 4:
            occupation_self_4 = 1
        elif occupation == 5:
            occupation_self_5 = 1
        elif occupation == 6:
            occupation_self_6 = 1

        

        occupation_husb_1 = 0
        occupation_husb_2 = 0
        occupation_husb_3 = 0
        occupation_husb_4 = 0
        occupation_husb_5 = 0
        occupation_husb_6 = 0
        if husbandOccupation == 1:
            occupation_husb_1 = 0
        elif husbandOccupation == 2:
            occupation_husb_2 = 1
        elif husbandOccupation == 3:
            occupation_husb_3 = 1
        elif husbandOccupation == 4:
            occupation_husb_4 = 1
        elif husbandOccupation == 5:
            occupation_husb_5 = 1
        elif husbandOccupation == 6:
            occupation_husb_6 = 1

        X = [[1,rateMarriage,age,yearsMarried,children,religious,educ,occupation_self_2,
        occupation_self_3,occupation_self_4,occupation_self_5,occupation_self_6,occupation_husb_2,
        occupation_husb_3,occupation_husb_4,occupation_husb_5,occupation_husb_6]]

        
        pred = reg.predict([[1,rateMarriage,age,yearsMarried,children,religious,educ,occupation_self_2,
        occupation_self_3,occupation_self_4,occupation_self_5,occupation_self_6,occupation_husb_2,
        occupation_husb_3,occupation_husb_4,occupation_husb_5,occupation_husb_6]])

        prediction=reg.predict_proba(X)
        output='{0:.{1}f}'.format(prediction[0][1], 2)

        if pred>0:
            return render_template('predict.html',response ='Your Marriage is in Danger')
        else:
            return render_template('predict.html',response ='Your Marriage is safe.')
    


   
        
        
if __name__ == '__main__':
    # importing models
    app.run(host='0.0.0.0', port=8001, debug=True)