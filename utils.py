from datetime import datetime, timedelta, timezone
import numpy as np


def datetime_to_seconds(dt):
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    seconds = int(delta.total_seconds())
    return seconds


def seconds_to_datetime(seconds):
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    dt = epoch + timedelta(seconds=seconds)
    return dt


def generate_data_record(device_id):
    daily_freq = 1 / (24 * 60 * 60)
    annual_freq = 1 / (365 * 24 * 60 * 60)

    timestamp = datetime_to_seconds(datetime.now(timezone.utc))

    daily_oscillation = np.sin(2 * np.pi * daily_freq * timestamp)
    annual_oscillation = np.sin(2 * np.pi * annual_freq * timestamp)

    temperature = 25 + 7 * (daily_oscillation + annual_oscillation) + np.random.normal()
    humidity = 10 + 5 * (daily_oscillation + annual_oscillation) + np.random.normal()

    return timestamp, device_id, float(temperature), float(humidity)
