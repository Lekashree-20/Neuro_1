# app.py

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import google.generativeai as gemini

import google.generativeai as genai
from flask_cors import CORS

# Configure the Gemma API
genai.configure(api_key="AIzaSyD1FPKl0lENNaIw8JGtMBzPXopVDIqcab8")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the saved model and scaler
with open('parkinsons_voting_clf.pkl', 'rb') as model_file:
    voting_clf, scaler, feature_names = pickle.load(model_file)

# Function to predict Parkinson's disease and risk score based on input data
def predict_disease_risk(input_data):
    input_data = input_data[feature_names]  # Ensure the input matches training features
    probabilities = voting_clf.predict_proba(input_data)
    predicted_label = voting_clf.predict(input_data)[0]
    
    # Decode the predicted disease (assuming binary classification for Parkinson's)
    predicted_disease = "Parkinson's Disease" if predicted_label == 1 else "No Parkinson's Disease"
    
    # Calculate the risk factor (as a percentage)
    predicted_risk = probabilities[0][predicted_label] * 100
    
    return predicted_disease, predicted_risk

# Function to generate a prevention report using the Gemma API
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
       - Purpose of the report
       - Context of general health and wellness

    2. **Risk Description**
       - General description of the identified risk
       - Common factors associated with the risk

    3. **Stage of Risk**
       - General information about the risk stage
       - Any typical considerations

    4. **Risk Assessment**
       - General overview of the risk's impact on health

    5. **Findings**
       - General wellness observations
       - Supporting information

    6. **Recommendations**
       - General wellness tips and lifestyle changes
       - Actions to promote well-being

    7. **Way Forward**
       - Suggested next steps for maintaining health
       - Advanced follow-up actions for this risk, like how we can overcome it.

    8. **Conclusion**
       - Summary of overall wellness advice
       - General support resources

    9. **Contact Information**
       - Information for general inquiries

    10. **References**
        - Simplified wellness resources (if applicable)

    **Details:**
    Risk: {risk}%
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "No content generated."
    except Exception as e:
        print(f"An error occurred during text generation: {e}")
        return None

# Combined route for predicting Parkinson's disease, calculating risk, and generating prevention report
@app.route('/predict_and_generate_report', methods=['POST'])
def predict_and_generate_report():
    try:
        # Get the input data from the request
        data = request.json
        
        # Debugging: Check if all feature names are present in the input data
        missing_features = [feature for feature in feature_names if feature not in data]
        if missing_features:
            return jsonify({'error': f"Missing features in input data: {', '.join(missing_features)}"})
        
        # Extract and validate input data
        input_data = [float(data[feature]) for feature in feature_names]

        # Convert the input data to a numpy array and reshape it for prediction
        input_data = np.asarray(input_data).reshape(1, -1)
        risks="Neurology"
        # Scale the input data
        std_data = scaler.transform(input_data)
        
        # Predict disease and risk
        disease, risk = predict_disease_risk(pd.DataFrame(std_data, columns=feature_names))
        
        response = {

            'risks':risks,
            'disease': disease,
            'risk score': f"{risk:.2f}%"
        }

        # Generate the prevention report if Parkinson's Disease is detected
        if "Parkinson's Disease" in disease:
            age = data.get('age', 50)  # Age can be dynamic; defaulting to 50 if not provided
            report = generate_prevention_report(risk, disease, age)
            response['report'] = report
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
