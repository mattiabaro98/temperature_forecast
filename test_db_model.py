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

def initial_training():
    end_time = datetime.now(timezone.utc) 
    start_time = end_time - timedelta(seconds=20000)

    data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
    data = data[-200:]
    temperatures = [t[2] for t in data]
    temperatures = np.array(temperatures).reshape(200,1)

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

def update_model():
    global temperature_forecast
    end_time = datetime.now(timezone.utc) 
    start_time = end_time - timedelta(seconds=10000)

    data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
    data = data[-120:]
    temperatures = [t[2] for t in data]
    temperatures = np.array(temperatures).reshape(120,1)

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

def run_periodic_task():
    while True:
        time.sleep(120)
        print("update the model")
        update_model()

def get_prediction():
    N_inferences = 5
    
    end_time = datetime.now(timezone.utc) 
    start_time = end_time - timedelta(seconds=2000)

    data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
    
    if len(data) < 60:
        raise ValueError("Not enough data")
    
    data = data[-60:]
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

with open('params.json') as file:
    params = json.load(file)
    print("create model")
    temperature_forecast = TemperatureForecast(params)
    print("perform initial training")
    initial_training()

def main():
    time.sleep(100)
    print(get_prediction())

if __name__ == "__main__":
    task_thread = threading.Thread(target=run_periodic_task)
    task_thread.start()
    main()
