import json
import numpy as np

import time
import threading

from flask import Flask, jsonify, request
from model import TemperatureForecast

import os
from dotenv import load_dotenv
from db_interaction import db_interaction
from datetime import datetime, timedelta, timezone
from utils import datetime_to_seconds

app = Flask(__name__)

IP = "0.0.0.0"

def get_continous_data(data:list) -> list:
    INTERVAL_TH = 10
    timestamps = [t[0] for t in data]
    temperatures = [t[2] for t in data]
    temperatures = np.array(temperatures).reshape(-1,1)

    continous_data_list = []
    initial_index = 0

    for i in range(1, len(timestamps)):
        current_integer = timestamps[i]
        previous_integer = timestamps[i - 1]
        
        if current_integer - previous_integer > INTERVAL_TH:
            continous_data_list.append(temperatures[initial_index:i])
            initial_index = i

    continous_data_list.append(temperatures[initial_index:])

    return continous_data_list

def get_training_data(start_time:datetime, end_time:datetime) -> tuple[np.ndarray, np.ndarray]:
    data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
    continous_data_list = get_continous_data(data)
    
    num_input_points = 60
    num_output_points = 15

    X = []
    Y = []

    for continous_data in continous_data_list:
        for i in range(continous_data.shape[0] - num_input_points - num_output_points):
            X.append(continous_data[i : i + num_input_points])
            Y.append(continous_data[i + num_input_points : i + num_input_points + num_output_points])

    X = np.array(X).reshape(-1, num_input_points, 1)
    Y = np.array(Y).reshape(-1, num_output_points)

    return X,Y

def initial_training():
    global temperature_forecast
    global last_timestamp
    
    end_time = datetime.now(timezone.utc) 
    start_time = end_time - timedelta(seconds=60*120)
    
    X,Y = get_training_data(start_time, end_time)
    
    if len(X) > 50:
        temperature_forecast.train_model(X,Y)
    
    last_timestamp = end_time

def update_model():
    global temperature_forecast
    global last_timestamp
    
    end_time = datetime.now(timezone.utc) 
    start_time = last_timestamp

    X,Y = get_training_data(start_time, end_time)
    
    if len(X) > 10:
        temperature_forecast.train_model(X,Y)
        temperature_forecast.update_inference_model()
    
    last_timestamp = end_time

def run_periodic_task():
    while True:
        time.sleep(120)
        update_model()

load_dotenv(override=True)

dbname = os.getenv("DB_NAME")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")

db_interactor = db_interaction(dbname=dbname, user=user, password=password, host=host, port=port)

last_timestamp = 0

with open('params.json') as file:
    params = json.load(file)
    temperature_forecast = TemperatureForecast(params)
    initial_training()

@app.route('/get_prediction', methods=['POST'])
def process_array():
    N_inferences = 5
    data = request.get_json(force=True)
    
    if 'temperatures' in data:
        temperatures = np.array(data['temperatures'])
        
        if temperatures.shape[0] == 60:
            total_temperatures = np.array(temperatures).reshape(-1,1)
            predicted_data = temperature_forecast.inference(total_temperatures).reshape(-1,1)
            total_temperatures = np.concatenate((total_temperatures, predicted_data), axis=0)
            total_predicted_data = predicted_data

            for i in range(N_inferences-1):
                predicted_data = temperature_forecast.inference(total_temperatures[-60:]).reshape(-1,1)
                total_temperatures = np.concatenate((total_temperatures, predicted_data), axis=0)
                total_predicted_data = np.concatenate((total_predicted_data, predicted_data), axis=0)
            
            return jsonify(total_predicted_data.tolist()), 200
        else:
            return jsonify({'error': 'The lenght of input data must be 60'}), 400

    else:
        return jsonify({'error': 'No array provided'}), 400

if __name__ == "__main__":
    task_thread = threading.Thread(target=run_periodic_task)
    task_thread.start()
    app.run(host="0.0.0.0", port=8088)