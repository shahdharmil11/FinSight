from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
from waitress import serve
import numpy as np
import joblib
import pandas as pd
import logging
from google.cloud import storage
import os

# Initialize the Flask application
app = Flask(__name__)

# Set up logging to debug level to capture all events
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

# Buckets Config
storage_client = storage.Client.from_service_account_json("/app/src/.google-auth.json")
bucket_name = "mlops_deploy_storage"
bucket = storage_client.bucket(bucket_name)

# Directory to store uploaded files for prediction
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Root endpoint which provides a basic greeting
@app.route('/')
def hello_world():
    return jsonify(message="Hello, World1!")

# Prediction endpoint which handles file uploads and makes predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file is part of the request
    if 'file' not in request.files:
        return jsonify(message='No file part'), 400

    file = request.files['file']
    # Check if the file has a name indicating that a file was indeed uploaded
    if file.filename == '':
        return jsonify(message='No selected file'), 400

    # Process only CSV files for prediction
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Define your parameters
        source_blob_name = 'model/production/scaler.joblib'
        destination_file_name = './model/scaler.joblib'

        # Download the model from GCS
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        # Load the data from the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
        scaler = joblib.load('./model/scaler.joblib')  # Load the scaler used for normalizing data
        df = np.array(df)
        df = df.reshape(-1, 1)  # Reshape data for scaling
        df = scaler.transform(df)  # Normalize data
        
        # Prepare data for prediction by creating sequences of 50 time steps
        x = []
        for i in range(50, df.shape[0]):
            x.append(df[i-50:i, 0])
        x = np.array(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # Reshape for model input

        # Define your parameters
        source_blob_name = 'model/production/stock_prediction.h5'
        destination_file_name = './model/stock_prediction.h5'

        # Download the model from GCS
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        # Load the model and make predictions
        model = load_model("./model/stock_prediction.h5")
        predictions = model.predict(x)
        predictions = scaler.inverse_transform(predictions)  # Reverse the scaling of predictions
        list_data = predictions.tolist()  # Convert predictions to list for JSON response

        return jsonify(list_data)

# Training endpoint (currently just a placeholder for future implementation)
@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()  # Get data from the POST request
    stock_ticker_symbol = data['stock_ticker_symbol']
    start_date = data['start_date']
    end_date = data['end_date']
    # Respond with a placeholder message
    return jsonify(message="Hello, Train!")

# Main function to run the app
if __name__ == '__main__':
    # Uncomment below to use waitress server for production
    # serve(app, host="0.0.0.0", port=5002, expose_tracebacks=True)
    app.run(debug=True, host="0.0.0.0", port=5002)  # Run the Flask app
