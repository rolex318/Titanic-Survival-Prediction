from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('models/model.pkl', 'rb'))

# Load scaler
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# =========================
# Home Route
# =========================

@app.route('/')
def home():
    return render_template('home.html')

# =========================
# Prediction Route
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get input values from form

        Pclass = int(request.form['Pclass'])
        Sex = int(request.form['Sex'])
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = int(request.form['Embarked'])

        # Create feature array

        features = np.array([[
            Pclass,
            Sex,
            Age,
            SibSp,
            Parch,
            Fare,
            Embarked
        ]])

        # Scale features

        features = scaler.transform(features)

        # Predict

        prediction = model.predict(features)

        # Output result

        if prediction[0] == 1:
            result = "Passenger Survived"
        else:
            result = "Passenger Did Not Survive"

        return render_template(
            'home.html',
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            'home.html',
            prediction_text=f"Error: {str(e)}"
        )

# =========================
# Main Function
# =========================

if __name__ == "__main__":
    app.run(debug=True)