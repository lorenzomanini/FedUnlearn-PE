class CommunicationTracker:
    """Preserve the legacy analytical communication accounting."""

    def __init__(self, num_clients: int, model):
        self.num_clients = num_clients
        self.communication_rounds = 0
        self.model_size_bytes = sum(p.numel() * 4 for p in model.parameters())
        self.num_parameters = sum(p.numel() for p in model.parameters())
        self.uplink_bytes_per_client = [0] * num_clients
        self.downlink_bytes_per_client = [0] * num_clients

    def record_round(self, participating_clients: list = None):
        self.communication_rounds += 1
        if participating_clients is None:
            participating_clients = list(range(self.num_clients))
        for client_id in participating_clients:
            self.uplink_bytes_per_client[client_id] += self.model_size_bytes
            self.downlink_bytes_per_client[client_id] += self.model_size_bytes

    def record_downlink(self, participating_clients: list = None):
        self.communication_rounds += 1
        if participating_clients is None:
            participating_clients = list(range(self.num_clients))
        for client_id in participating_clients:
            self.downlink_bytes_per_client[client_id] += self.model_size_bytes

    def get_metrics(self) -> dict:
        return {
            "communication_rounds": self.communication_rounds,
            "num_parameters": self.num_parameters,
            "model_size_bytes": self.model_size_bytes,
            "uplink_bytes_per_client": self.uplink_bytes_per_client.copy(),
            "downlink_bytes_per_client": self.downlink_bytes_per_client.copy(),
            "total_uplink_bytes": sum(self.uplink_bytes_per_client),
            "total_downlink_bytes": sum(self.downlink_bytes_per_client),
            "total_communication_bytes": sum(self.uplink_bytes_per_client)
            + sum(self.downlink_bytes_per_client),
            "avg_uplink_bytes_per_client": sum(self.uplink_bytes_per_client) / self.num_clients
            if self.num_clients > 0
            else 0,
            "avg_downlink_bytes_per_client": sum(self.downlink_bytes_per_client) / self.num_clients
            if self.num_clients > 0
            else 0,
        }

    def __repr__(self):
        metrics = self.get_metrics()
        return (
            f"CommunicationTracker(rounds={metrics['communication_rounds']}, "
            f"total_bytes={metrics['total_communication_bytes']:,})"
        )
