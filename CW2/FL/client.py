import os
import flwr as fl
import pandas as pd
from model_loader import preprocess_data_CNN, CNN_model, preprocess_data_LSTM, LSTM_model


class Client(fl.client.NumPyClient):
    def __init__(self, df):
        self.X_train, self.X_test, self.Y_train, self.Y_test = preprocess_data_CNN(df)  # get splited data
        self.model = CNN_model(self.X_train, self.Y_train)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.Y_train, epochs=2, batch_size=16)
        return self.model.get_weights(), len(self.X_train), {k: v[-1] for k, v in history.history.items()}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    datapath = (os.path.join(os.getcwd(), 'CW1', 'datasets', 'filtered_df_reduced.csv'))
    df = pd.read_csv(datapath)

    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    fl.client.start_client(server_address=server_address, client=Client(df).to_client())