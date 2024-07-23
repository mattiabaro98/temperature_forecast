import io
import json
import logging
import zipfile

import numpy as np
import requests

# import google.cloud.logging
from flask import Flask, jsonify, request
from model import TemperatureForecast

app = Flask(__name__)

IP = "0.0.0.0"

@app.route("/get_inference", methods=["POST"])
def get_inference():
    logging.info("begin of method")
    params = request.get_json()
    
    # Check if there is data
    if params is None:
        return jsonify({'error': 'No JSON data provided'}), 400

    # Process the JSON data (Example: print it or perform operations)
    print(params)

    temperature_forecast = TemperatureForecast(params=params)
    
    logging.info("Correctly create the model")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088)