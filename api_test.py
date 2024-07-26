import os
import requests
import numpy as np
from dotenv import load_dotenv
from db_interaction import db_interaction
from datetime import datetime, timedelta, timezone
from utils import datetime_to_seconds

def get_data_from_db(db_interactor, N:int) -> np.ndarray:
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

def main():
    
    load_dotenv(override=True)

    dbname = os.getenv("DB_NAME")
    user = os.getenv("USER")
    password = os.getenv("PASSWORD")
    host = os.getenv("HOST")
    port = os.getenv("PORT")

    db_interactor = db_interaction(dbname=dbname, user=user, password=password, host=host, port=port)

    temperatures = get_data_from_db(db_interactor, 60)
    data = {'temperatures': temperatures.tolist()}

    url = "http://0.0.0.0:8088/get_prediction"

    print("Sending data to server...")

    response = requests.post(url, json=data)
    
    if response.status_code == 200:
         array_data = response.json()
         np_array = np.array(array_data)
         print('Received ndarray:\n', np_array)
    else:
         print('Failed to get array from the API. Status code:', response.status_code)

if __name__ == "__main__":
    main()