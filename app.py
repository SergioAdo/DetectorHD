from flask import Flask, request, jsonify, render_template
from pycaret.classification import *
import pandas as pd

app = Flask(__name__)
model = load_model('models/model_rf')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    tmp= pd.read_csv("tmp.csv")
    inputs =[e for e in request.form.values()]
    inputs = inputs[1:14]
    tmp['age'] = inputs[0]
    tmp['sex'] = inputs[1]
    tmp['chest pain type'] = inputs[2]
    tmp['resting bp s']= inputs[3]
    tmp['cholesterol'] = inputs[4]
    tmp['fasting blood sugar'] = inputs[5]
    tmp['resting ecg'] = inputs[6]
    tmp['max heart rate'] = inputs[7]
    tmp['exercise angina'] = inputs[8]
    tmp['oldpeak'] = inputs[9]
    tmp['ST slope'] = inputs[10]


    pred_tmp= predict_model(model, data= tmp)
    score= 100*(pred_tmp.Score[0])
    print(score)

    if (pred_tmp['Label'] == 0).any():
        if inputs[1]== "0":
            return render_template('results.html', prediction_text= "Vous êtes saine Madame avec un score de {}".format(score) + "%" )
        else:
                    return render_template('results.html', prediction_text= "Vous êtes sain Monsieur avec un score de {}".format(score) + "%" )
    else:
        if inputs[1]== "0":
           return render_template('results.html', prediction_text= "Vous êtes malade Madame avec un score de {}".format(score) + "%" )
        else:
           return render_template('results.html', prediction_text= "Vous êtes malade Monsieur avec un score de {}".format(score) + "%" )
    return render_template('results.html', prediction_text= pred_tmp)

if __name__ == "__main__":
    app.run(debug=True)