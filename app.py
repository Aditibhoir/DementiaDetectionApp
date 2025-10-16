from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


# Load the trained model
model = pickle.load(open('dementia_model.pkl', 'rb'))

# List of all features in the trained model
features = [
    'Age', 'Gender', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
    'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
    'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion', 'Disorientation',
    'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
]

@app.route('/')
def index():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    # Read all feature inputs from form and convert to numeric
    input_data = []
    for feature in features:
        value = request.form.get(feature)
        if value is None or value == '':
            value = 0
        input_data.append(float(value))
    
    # Make prediction
    prediction = model.predict([input_data])[0]
    probabilities = model.predict_proba([input_data])[0]
    
    # Probability for Dementia
    dementia_prob = probabilities[1] * 100  # 0=No Dementia, 1=Dementia
    
    if prediction == 1:
        result_text = f"🧠 The person is likely showing signs of Dementia. (Confidence: {dementia_prob:.2f}%)"
    else:
        result_text = f"✅ The person is not showing signs of Dementia. (Confidence: {100 - dementia_prob:.2f}%)"
    
    return render_template('result.html', result=result_text, dementia_prob=dementia_prob)

    return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)

