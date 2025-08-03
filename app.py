from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your scaler and model (make sure these files exist)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_input(data):
    # Convert categorical variables to numeric (example)
    sex = 1 if data['Sex'] == 'male' else 0
    intensity_map = {'Low': 0, 'Medium': 1, 'High': 2}
    intensity = intensity_map.get(data['Intensity'], 0)

    features = [
        sex,
        intensity,
        float(data['Age']),
        float(data['Height']),
        float(data['Weight']),
        float(data['Duration']),
        float(data['Heart_Rate']),
        float(data['Body_Temp'])
    ]

    # Scale features
    features_scaled = scaler.transform([features])
    return features_scaled

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    form_data = None

    if request.method == 'POST':
        form_data = request.form
        try:
            X = preprocess_input(form_data)
            pred = model.predict(X)[0]
            prediction = round(pred, 2)
        except Exception as e:
            error = str(e)

    return render_template('index.html', prediction=prediction, error=error, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
