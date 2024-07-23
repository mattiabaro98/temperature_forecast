import json
import numpy as np
import requests

import time
import threading

# import google.cloud.logging
from flask import Flask, jsonify, request
from model import TemperatureForecast

app = Flask(__name__)

IP = "0.0.0.0"

with open('params.json') as file:
    params = json.load(file)
    temperature_forecast = TemperatureForecast(params)

def update_model():
    global temperature_forecast
    temperature_forecast.update_inference_model()

def run_periodic_task():
    while True:
        update_model()
        time.sleep(120)

@app.route("/get_inference", methods=["POST"])
def get_inference():
    params = request.get_json()
    
    response = {
        "received_data": params,
        "message": "Inference received successfully"
    }

    global temperature_forecast
    temperature_forecast = TemperatureForecast(params=params)
    
    return jsonify(response), 200

    # # Check if there is data
    # if params is None:
    #     return jsonify({'error': 'No JSON data provided'}), 400

    # # Process the JSON data (Example: print it or perform operations)
    # print(params)

    # temperature_forecast = TemperatureForecast(params=params)
    
    # logging.info("Correctly create the model")

    # response = {"model": "created"}

    # return jsonify(response)

@app.route("/get_prediction", methods=["GET"])
def get_prediction():
    return temperature_forecast

if __name__ == "__main__":
    task_thread = threading.Thread(target=run_periodic_task)
    task_thread.start()
    app.run(host="0.0.0.0", port=8088)