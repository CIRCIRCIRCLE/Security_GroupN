import flwr as fl
import os
import sys
from client import Client
from server import get_server_strategy 
from functools import partial
import numpy as np
import pandas as pd

def split_dataset_for_clients(df, num_clients):
    partition_size = df.shape[0] // num_clients
    print('The partition size is: ', partition_size)

    # Split the data into partitions for each client
    dfs = np.array_split(df, num_clients)
    client_dfs = {}

    # Assign each partition to a client
    for idx, client_df in enumerate(dfs):
        client_dfs[str(idx)] = client_df
    return client_dfs


def create_client(cid, client_dfs):
    return Client(client_dfs[cid])

if __name__ == "__main__":
    # load original dataset
    datapath = (os.path.join(os.getcwd(), 'CW1', 'datasets', 'filtered_df_reduced.csv'))
    df = pd.read_csv(datapath)

    #partition the dataset
    NUM_CLIENTS = 3
    client_dfs = split_dataset_for_clients(df, NUM_CLIENTS)
    for client_id in range(NUM_CLIENTS):
        print(f"Client {client_id}:")
        print(f"Client dfs: {client_dfs[str(client_id)].shape}")
        print()
    client_fnc = partial(create_client, client_dfs)
    print(client_fnc)

    history = fl.simulation.start_simulation(
        client_fn=client_fnc,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=get_server_strategy(),
        client_resources={"num_cpus": 1, "num_gpus": 0},
        ray_init_args={
            "num_cpus": 1,
            "num_gpus": 0,
            "_system_config": {"automatic_object_spilling_enabled": False},
        },
    )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")