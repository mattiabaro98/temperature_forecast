import numpy as np
import pandas as pd

time_number = 30
time_unit = "s"
num_points = int(30 * 24 * 3600 / time_number)

start_datetime = np.datetime64("2024-07-01")
timestamp = np.array([start_datetime + n * np.timedelta64(time_number, time_unit) for n in range(num_points)])

np.random.seed(0)
t = np.linspace(0, 10, num_points)

freq1 = 1
freq2 = 0.5
freq3 = 0.25

sine_wave1 = np.sin(2 * np.pi * freq1 * t)
sine_wave2 = np.sin(2 * np.pi * freq2 * t)
sine_wave3 = np.sin(2 * np.pi * freq3 * t)

temperature = sine_wave1 + sine_wave2 + sine_wave3

noise = 0.1 * np.random.normal(size=num_points)
temperature_with_noise = temperature + noise
temperature_with_noise = 28 + temperature_with_noise

pd.DataFrame(
    {
        "timestamp": timestamp,
        "temperature": temperature_with_noise,
    }
).to_csv("dataset.csv", index=False)
