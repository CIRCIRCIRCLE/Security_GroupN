import os
import pandas as pd
import flwr as fl
from model_loader import preprocess_data_CNN, CNN_model
from flwr.client import NumPyClient, ClientApp
from flwr.client.mod.localdp_mod import LocalDpMod

# Define Differential Privacy Configuration
class DpConfig:
    clipping_norm = 1.0  # Maximum allowed L2 norm of the gradients
    sensitivity = 1.0    # Sensitivity of the query
    epsilon = 1         # Privacy budget
    delta = 1e-5         # Probability of privacy breach

# Create an instance of LocalDpMod with the required parameters
local_dp_mod = LocalDpMod(
    DpConfig.clipping_norm,
    DpConfig.sensitivity,
    DpConfig.epsilon,
    DpConfig.delta
)

# Flower client class definition
class FlowerClient(NumPyClient):
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

def client_fn() -> FlowerClient:
    """Function to load data and return an instance of FlowerClient."""
    datapath = os.path.join(os.getcwd(), 'CW2', 'datasets', 'filtered_df_reduced.csv')
    df = pd.read_csv(datapath)
    return FlowerClient(df).to_client()

    
# Initialize and start the client application with differential privacy
app = ClientApp(
    client_fn=client_fn,
    mods=[local_dp_mod],
)

# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=client_fn(),
    )
