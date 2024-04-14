import flwr as fl
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}

def get_server_strategy():
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 2 clients for training
        min_evaluate_clients = 2,  # Never sample less than 5 clients for evaluation
        min_available_clients = 2, # Wait until all 5 clients are available
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        )

# Define config
config = ServerConfig(num_rounds=1)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=get_server_strategy(),
)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    history = start_server(
                server_address="0.0.0.0:8080",
                config=config,
                strategy=get_server_strategy(),
            )
    final_round, acc = history.metrics_distributed["accuracy"][-1]
    print(f"After {final_round} rounds of training the accuracy is {acc:.3%}")

