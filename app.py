import flask
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (update to use the correct model file)
model = joblib.load('C:/Users/jesus/OneDrive/Desktop/inventory demand forecasting/model.pkl')  # Use your existing model file

# Load dataset (if needed for making predictions or validating inputs)
df = pd.read_csv('retail_store_inventory.csv')

@app.route('/')
def home():
    return "Welcome to the Inventory Demand Forecasting Model API!"

# Prediction Route for Model
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data.get('input')

    if input_data is None:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Predict using the model
        prediction = model.predict([input_data])  # Predict based on input data
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
