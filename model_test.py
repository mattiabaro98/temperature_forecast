import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Tuple

from model import TemperatureForecast

def get_inference_data(
    temperatures_datsframe: pd.DataFrame, inference_from: datetime = datetime.now(), num_input_points: int = 60
) -> Tuple[np.ndarray, np.ndarray]:

    closest_date = min(temperatures_datsframe["timestamp"], key=lambda x: abs(x - inference_from))
    closest_date_index = temperatures_datsframe[temperatures_datsframe["timestamp"] == closest_date].index.tolist()[0]
    temperatures = temperatures_datsframe["temperature"].to_numpy()

    return temperatures[closest_date_index - num_input_points : closest_date_index].reshape(-1, 1)


def get_training_set(
    temperatures_datsframe: pd.DataFrame,
    train_from: datetime = datetime.now(),
    train_for: int = 3600 * 24,
    num_input_points: int = 60,
    num_output_points: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:

    closest_date = min(temperatures_datsframe["timestamp"], key=lambda x: abs(x - train_from))
    start_date = closest_date - timedelta(seconds=train_for)
    temperatures_datsframe = temperatures_datsframe[
        temperatures_datsframe["timestamp"].between(start_date, closest_date)
    ]
    temperatures = temperatures_datsframe["temperature"].to_numpy()
    
    X = []
    Y = []

    for i in range(temperatures.shape[0] - num_input_points - num_output_points):
        X.append(temperatures[i : i + num_input_points])
        Y.append(temperatures[i + num_input_points : i + num_input_points + num_output_points])

    X = np.array(X).reshape(-1, num_input_points, 1)
    Y = np.array(Y).reshape(-1, num_output_points)

    return X, Y

def main():
    temperatures_datsframe = pd.read_csv("./dataset.csv")
    temperatures_datsframe["timestamp"] = pd.to_datetime(temperatures_datsframe["timestamp"])

    with open("./params.json", "r") as file:
        params = json.load(file)

    temperature_forecast = TemperatureForecast(params=params)

    num_input_points = temperature_forecast.get_num_input_points()
    num_output_points = temperature_forecast.get_num_output_points()

    train_from = datetime.now()
    train_for = 3600 * 24
    X, Y = get_training_set(temperatures_datsframe, train_from, train_for, num_input_points, num_output_points)
    temperature_forecast.train_model(X, Y)
    path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".weights.h5"
    temperature_forecast.save_model_weights(path)

    temperature_forecast.update_inference_model()

    inference_from = datetime.now()

    X = get_inference_data(temperatures_datsframe, inference_from, num_input_points).reshape(1, -1, 1)
    print(X.shape)
    Y = temperature_forecast.inference(X)
    print(Y.shape)

    train_from = datetime.now()
    train_for = 3600 * 24
    X, Y = get_training_set(temperatures_datsframe, train_from, train_for, num_input_points, num_output_points)
    print(X.shape, Y.shape)
    test_loss = temperature_forecast.evaluate(X, Y)
    print(test_loss)

if __name__ == "__main__":
    main()