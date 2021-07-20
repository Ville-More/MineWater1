import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

# Create a Flask application

app = Flask(__name__)

# Load pickle model

model = pickle.load(open('Final_MultiOutputRF_model.pkl', 'rb'))

# Create a home page

@app.route('/')
def home():
    return render_template('index.html')

# Create a POST method

@app.route('/predict', methods = ['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text = 'EC & pH should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug = True)

