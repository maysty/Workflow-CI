from prometheus_client import start_http_server, Counter, Gauge
import time

requests_total = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

errors_total = Counter(
    "prediction_errors_total",
    "Total prediction errors"
)

latency = Gauge(
    "prediction_latency_seconds",
    "Prediction latency"
)

if __name__ == "__main__":
    start_http_server(8001)
    print("Exporter running on http://localhost:8001/metrics")

    while True:
        requests_total.inc()
        latency.set(0.2)
        time.sleep(5)