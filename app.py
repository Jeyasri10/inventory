import flask
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')
model3 = joblib.load('model3.pkl')
model4 = joblib.load('model4.pkl')
model5 = joblib.load('model5.pkl')

# Load dataset (if needed for making predictions or validating inputs)
df = pd.read_csv('retail_store_inventory.csv')

@app.route('/')
def home():
    return "Welcome to the Inventory Demand Forecasting Model API!"

# Model 1 Prediction Route
@app.route('/predict/model1', methods=['POST'])
def predict_model1():
    data = request.get_json()
    input_data = data.get('input')

    if input_data is None:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        prediction = model1.predict([input_data])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model 2 Prediction Route
@app.route('/predict/model2', methods=['POST'])
def predict_model2():
    data = request.get_json()
    input_data = data.get('input')

    if input_data is None:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        prediction = model2.predict([input_data])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model 3 Prediction Route
@app.route('/predict/model3', methods=['POST'])
def predict_model3():
    data = request.get_json()
    input_data = data.get('input')

    if input_data is None:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        prediction = model3.predict([input_data])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model 4 Prediction Route
@app.route('/predict/model4', methods=['POST'])
def predict_model4():
    data = request.get_json()
    input_data = data.get('input')

    if input_data is None:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        prediction = model4.predict([input_data])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model 5 Prediction Route
@app.route('/predict/model5', methods=['POST'])
def predict_model5():
    data = request.get_json()
    input_data = data.get('input')

    if input_data is None:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        prediction = model5.predict([input_data])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
