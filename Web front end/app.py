from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
from flask import Flask, url_for
app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load("C:/Users/marti/Martina/riskfactor.joblib")

# Create a scaler object
scaler = StandardScaler()
app = Flask(__name__, static_folder='static')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/risk_factor')
def risk_factor():
    return render_template('risk_factors.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict adsdasdsadasdasdasdasd.....!")
    # sys.stdout.close()
    # sys.stdout = sys.__stdout__
    # Get the input data from the request
    print("Predict .....!")
    #print(request.form['question2'])
    #print(request.form['q2'])
    #print(request.form['q3'])
    #print(request.form['q4'])
    #input_data = request.get_json()
    age = int(request.form['q1'])
    print(age)
    noSexPart = request.form['q2']
    firstSexInt = int(request.form['q3'])
    pregnant = int(request.form['q4'])
    smoke = request.form['q5']
    how_many_smoke = int(request.form['q6'])
    hormonalContra = request.form['q7']
    hormonalContra_years = int(request.form['q8'])
    IUD = request.form['q9']
    IUD_years = int(request.form['q10'])
    STDs = request.form['q11']
    STDs_num =  int(request.form['q12'])
    condylomatosis = request.form['q13']
    cervical_condylomatosis = request.form['q14']
    vaginal_condylomatosis = request.form['q15']
    vulvo_perineal_condylomatosis = request.form['q16']
    syphilis = request.form['q17']
    PID = request.form['q18']
    herpes = request.form['q19']
    molluscum = request.form['q20']
    AIDS = request.form['q21']
    HIV = request.form['q22']
    hepatitisB = request.form['q23']
    HPV = request.form['q24']
    STDs_times = int(request.form['q25'])
    STDs_first = int(request.form['q26'])
    STDs_last = int(request.form['q27'])
    cervical_cancer =  request.form['q28']
    CIN =  request.form['q29']
    HPV_diagnosed =  request.form['q30']
    other_disease =  request.form['q31']
    other_disease_num = int(request.form['q32'])
    HIV_diagnosed =  request.form['q33']


    # Convert the input data to a numpy array
    input_array = np.array([[age,noSexPart,firstSexInt,pregnant,smoke,how_many_smoke, hormonalContra,hormonalContra_years,IUD , IUD_years,STDs,STDs_num,condylomatosis,cervical_condylomatosis,vaginal_condylomatosis,vulvo_perineal_condylomatosis,syphilis,PID, herpes, molluscum,AIDS,HIV,hepatitisB,HPV,STDs_times,STDs_first,STDs_last, cervical_cancer,CIN,HPV_diagnosed,other_disease,other_disease_num, HIV_diagnosed]])
    input_array = input_array.astype(np.int32)
    y = loaded_model.predict(input_array)
    print(input_array)
    print(y)

    if y[0] == 0:
         return render_template('healthy.html')
        #return "You are not at risk of Cervical Cancer"
    else:
        #return "You are at risk of cervical cancer, Follow-Up is recommended"
        return render_template('haveCancer.html')
    

@app.route('/help_yourself')
def help_yourself():
    app.logger.error('An errorsss occurred')
    return render_template('help_yourself.html')

@app.route('/signs_symptoms')
def signs_symptoms():
    return render_template('signs & symptoms.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')


if __name__ == '__main__':
    #print("aaa")
    app.run(debug=True)