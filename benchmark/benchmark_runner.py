# benchmark/benchmark_runner.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.load_model import load_model_and_input
from utils.profiler import profile_model_run
import json
import os

def run_benchmark():
    model, input_tensor = load_model_and_input()
    metrics = profile_model_run(model, input_tensor)

    os.makedirs("results", exist_ok=True)
    with open("results/metrics_log.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Benchmark complete. Results saved to results/metrics_log.json")

if __name__ == "__main__":
    run_benchmark()
