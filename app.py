import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Education_2 = Education_3 = Securities_Account_0 = Securities_Account_1 = CD_Account_0 = CD_Account_1 = Online_0 = Online_1 = CreditCard_0 = CreditCard_1 = 0
    Age = int(request.form['age'])
    Experience = int(request.form['exp'])
    Income = int(request.form['inc'])
    Family = int(request.form['FS'])
    CCAvg = float(request.form['ccavg'])
    Mortgage = int(request.form['mor'])
    Education = request.form['Education']
    if (Education == 1):
        Education_2 = 0
        Education_3 = 0
    elif (Education == 2):
        Education_2 = 1
        Education_3 = 0
    elif (Education == 3):
        Education_2 = 0
        Education_3 = 1
    Securities_Account = request.form['Securities_Account']
    if (Securities_Account == 1):
        Securities_Account_1 = 1
    elif (Securities_Account == 0):
        Securities_Account_1 = 0
    CD_Account = request.form['CD_Account']
    if ( CD_Account == 1):
        CD_Account_1 = 1
    elif (CD_Account == 0):
        CD_Account_1 = 0
    Online = request.form['Online']
    if (Online == 1):
        Online_1 = 1
    elif (Online == 0):
        Online_1 = 0
    CreditCard = request.form['CreditCard']
    if (CreditCard == 1):
        CreditCard_1 = 1
    elif (CreditCard == 0):
        CreditCard_1 = 0
    features_2 = [Age, Experience, Income, Family, CCAvg, Mortgage, Education_2,Education_3,Securities_Account_1,CD_Account_1,Online_1,CreditCard_1]
    final_features = [np.array(features_2)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    if output == 1:
        output = "Eligible"
    elif output == 0:
        output = "Not Eligible"

    return render_template('index.html', prediction_text='{} for Personal Loan'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
