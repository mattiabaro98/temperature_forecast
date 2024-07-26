import os
from dotenv import load_dotenv
from db_interaction import db_interaction
from datetime import datetime, timedelta, timezone
from utils import datetime_to_seconds
import numpy as np

def get_data_from_db(N:int) -> np.ndarray:
    total_data = []
    sec = N

    while len(total_data) < N:
        end_time = datetime.now(timezone.utc) 
        start_time = end_time - timedelta(seconds=sec)
        data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
        total_data = total_data + data
        sec = N - len(total_data)
    
    temperatures = [t[2] for t in total_data]
    temperatures = np.array(temperatures).reshape(N,1)

    return temperatures

load_dotenv(override=True)

dbname = os.getenv("DB_NAME")
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")
port = os.getenv("PORT")

db_interactor = db_interaction(dbname=dbname, user=user, password=password, host=host, port=port)

end_time = datetime.now(timezone.utc) 
start_time = end_time - timedelta(seconds=10000)

data = db_interactor.read_record_data(datetime_to_seconds(start_time), datetime_to_seconds(end_time))
selected_data = data[-120:]
temperatures = [t[2] for t in selected_data]
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
