import os
from dotenv import load_dotenv
from db_interaction import db_interaction
from datetime import datetime, timedelta, timezone
from utils import datetime_to_seconds
import numpy as np
import pandas as pd

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
data_array = np.array(data)
print(data_array.shape)
df = pd.DataFrame(data_array, columns=["timestamp", "device", "temperature", "umidity"])
print(df)