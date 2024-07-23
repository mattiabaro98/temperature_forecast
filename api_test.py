import argparse
import json

import requests

parser = argparse.ArgumentParser(description="Read ply and landmarks file.")
parser.add_argument("--path", type=str, help="Path to ply file.")
args = parser.parse_args()


def main():
    
    with open("./params.json", "r") as file:
        params = json.load(file)

    url = "http://0.0.0.0:8088/get_inference"

    response = requests.post(url, data=json.dumps(params))
    

if __name__ == "__main__":
    main()