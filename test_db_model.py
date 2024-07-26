import os
from dotenv import load_dotenv
from db_interaction import db_interaction
from datetime import datetime, timedelta, timezone
from utils import datetime_to_seconds
import numpy as np
import json
import time
import threading
from model import TemperatureForecast

def find_discontinuities(arr):
    discontinuities = []
    
    for i in range(1, len(arr)):
        current_integer = arr[i]
        previous_integer = arr[i - 1]
        
        if current_integer - previous_integer > 1:
            gap_start = previous_integer + 1
            gap_end = current_integer - 1
            discontinuities.append(gap_end-gap_start)
    
    return discontinuities

def check_data_continuity(data):
    interval_th = 10
    timestamps = [t[0] for t in data]
    discontinuities = find_discontinuities(timestamps)
    return all(x <= interval_th for x in discontinuities)

def check_data_quantity(data):
    return len(data) > 60

def initial_training():
    global temperature_forecast
    global last_timestamp
    end_time = datetime.now(timezone.utc) 
    start_time = end_time - timedelta(seconds=200)
    
    data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
    
    if check_data_continuity(data) and check_data_quantity(data):
    
        temperatures = [t[2] for t in data]
        temperatures = np.array(temperatures).reshape(-1,1)

        num_input_points = 60
        num_output_points = 15

        X = []
        Y = []

        for i in range(temperatures.shape[0] - num_input_points - num_output_points):
            X.append(temperatures[i : i + num_input_points])
            Y.append(temperatures[i + num_input_points : i + num_input_points + num_output_points])

        X = np.array(X).reshape(-1, num_input_points, 1)
        Y = np.array(Y).reshape(-1, num_output_points)

        temperature_forecast.train_model(X,Y)
    
    last_timestamp = datetime_to_seconds(end_time)
    
def update_model():
    global temperature_forecast
    global last_timestamp
    end_time = datetime.now(timezone.utc) 
    start_time = end_time - timedelta(seconds=last_timestamp)

    data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
    
    if check_data_continuity(data) and check_data_quantity(data):

        temperatures = [t[2] for t in data]
        temperatures = np.array(temperatures).reshape(-1,1)

        num_input_points = 60
        num_output_points = 15

        X = []
        Y = []

        for i in range(temperatures.shape[0] - num_input_points - num_output_points):
            X.append(temperatures[i : i + num_input_points])
            Y.append(temperatures[i + num_input_points : i + num_input_points + num_output_points])

        X = np.array(X).reshape(-1, num_input_points, 1)
        Y = np.array(Y).reshape(-1, num_output_points)

        temperature_forecast.train_model(X,Y)
        temperature_forecast.update_inference_model()
    
    last_timestamp = datetime_to_seconds(end_time)

def run_periodic_task():
    while True:
        time.sleep(120)
        print("update the model")
        update_model()

def get_prediction(data):
    N_inferences = 5
    
    temperatures = [t[2] for t in data]
    total_temperatures = np.array(temperatures).reshape(-1,1)
    predicted_data = temperature_forecast.inference(total_temperatures.reshape(1, -1, 1))
    predicted_data = predicted_data.reshape(-1,1)
    total_temperatures = np.concatenate((total_temperatures, predicted_data), axis=0)
    total_predicted_data = predicted_data

    for i in range(N_inferences-1):
        predicted_data = temperature_forecast.inference(total_temperatures[-60:].reshape(1, -1, 1))
        predicted_data = predicted_data.reshape(-1,1)
        total_temperatures = np.concatenate((total_temperatures, predicted_data), axis=0)
        total_predicted_data = np.concatenate((total_predicted_data, predicted_data), axis=0)
    
    return total_predicted_data


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
    print("create model")
    temperature_forecast = TemperatureForecast(params)
    print("perform initial training")
    initial_training()

def main():
    time.sleep(100)

    end_time = datetime.now(timezone.utc) 
    start_time = end_time - timedelta(seconds=2000)

    data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
    
    if len(data) < 60:
        raise ValueError("Not enough data")
    
    data = data[-60:]

    print(get_prediction(data))

if __name__ == "__main__":
    task_thread = threading.Thread(target=run_periodic_task)
    task_thread.start()
    main()
