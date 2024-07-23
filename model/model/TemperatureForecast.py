import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

class TemperatureForecast:

    def __init__(self, params: dict = None):

        self.num_input_points = params["num_input_points"]
        self.num_output_points = params["num_output_points"]
        self.model_params = params["model_params"]
        self.expected_temp_range = params["expected_temp_range"]
        self.normalization_range = params["normalization_range"]

        self.model_inference = self.create_model()
        self.model_training = self.create_model()

        self.updating = False
        self.training_loss = None

    def get_num_input_points(self) -> int:
        return self.num_input_points

    def get_num_output_points(self) -> int:
        return self.num_output_points

    def scale_data(self, array: np.ndarray) -> np.ndarray:
        return (max(self.normalization_range) - min(self.normalization_range)) * (
            array - min(self.expected_temp_range)
        ) / (max(self.expected_temp_range) - min(self.expected_temp_range)) + min(self.normalization_range)

    def inverse_scale_data(self, array: np.ndarray) -> np.ndarray:
        return (array - min(self.normalization_range)) * (
            max(self.expected_temp_range) - min(self.expected_temp_range)
        ) / (max(self.normalization_range) - min(self.normalization_range)) + min(self.expected_temp_range)

    def create_model(self) -> tf.keras.models.Sequential:

        model = Sequential()
        model.add(Input(shape=(self.num_input_points, 1)))
        model.add(LSTM(self.model_params["LSTM_num"], return_sequences=False))
        model.add(Dense(self.num_output_points))
        model.compile(
            optimizer=self.model_params["optimizer"],
            loss=self.model_params["loss"],
            metrics=self.model_params["metrics"],
        )

        return model

    def train_model(self, X: np.ndarray, Y: np.ndarray) -> tf.keras.models.Sequential:

        X = self.scale_data(X)
        Y = self.scale_data(Y)

        self.model_training.fit(
            X,
            Y,
            epochs=self.model_params["epochs"],
            batch_size=self.model_params["batch"],
            shuffle=True,
            verbose=0,
        )

        self.training_loss = self.model_training.evaluate(X, Y)

    def save_model_weights(self, weights_path: str):
        self.model_training.save_weights(weights_path)

    def read_model_weights(self, weights_path: str):
        self.model_training.load_weights(weights_path)

    def update_inference_model(self):
        self.updating = True
        self.model_inference = tf.keras.models.clone_model(self.model_training)
        self.model_inference.compile(
            optimizer=self.model_params["optimizer"],
            loss=self.model_params["loss"],
            metrics=self.model_params["metrics"],
        )
        self.updating = False

    def inference(self, X: np.ndarray) -> np.ndarray:
        X = self.scale_data(X)
        Y = self.model_inference.predict(X)
        Y = self.inverse_scale_data(Y)
        return Y

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> float:
        X = self.scale_data(X)
        Y = self.scale_data(Y)
        return self.model_inference.evaluate(X, Y)