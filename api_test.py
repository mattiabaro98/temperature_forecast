import argparse
import json

import requests
import numpy as np

parser = argparse.ArgumentParser(description="Read ply and landmarks file.")
parser.add_argument("--path", type=str, help="Path to ply file.")
args = parser.parse_args()


def main():
    
    with open("./params.json", "r") as file:
        params = json.load(file)

    url = "http://0.0.0.0:8088/get_prediction"

    response = requests.get(url)
    
    if response.status_code == 200:
         array_data = response.json()
         np_array = np.array(array_data)
         print('Received ndarray:\n', np_array)
    else:
         print('Failed to get array from the API. Status code:', response.status_code)

if __name__ == "__main__":
    main()