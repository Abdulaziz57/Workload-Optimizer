# benchmark/benchmark_runner.py
import os
import sys
import argparse
import json
from datetime import datetime

# Ensure Python can find your modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.load_model import load_model_and_input
from utils.profiler import profile_model_run

def parse_args():
    parser = argparse.ArgumentParser(description="Single-run benchmark")
    parser.add_argument("--model", type=str, default="mobilenet_v2",
                        choices=["mobilenet_v2", "resnet50", "vgg16", "bert"],
                        help="Model architecture to benchmark")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for input")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs to average")
    parser.add_argument("--use_half", action="store_true",
                        help="Use half-precision (fp16) on GPU if supported (ignored on CPU/MPS).")
    return parser.parse_args()

def run_benchmark():
    args = parse_args()

    # Load model & dummy input
    model, input_tensor, device_str = load_model_and_input(
        model_name=args.model,
        batch_size=args.batch_size,
        use_half=args.use_half
    )

    # Profile multiple runs
    times = []
    gpu_mem_diffs = []
    cpu_mem_diffs = []

    for _ in range(args.num_runs):
        metrics = profile_model_run(model, input_tensor, device_str)
        times.append(metrics["exec_time_sec"])
        gpu_mem_diffs.append(metrics["gpu_memory_diff_MB"])
        cpu_mem_diffs.append(metrics["cpu_memory_diff_MB"])

    # Compute average metrics
    avg_time = sum(times)/len(times)
    avg_gpu_mem = sum(gpu_mem_diffs)/len(gpu_mem_diffs)
    avg_cpu_mem = sum(cpu_mem_diffs)/len(cpu_mem_diffs)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "batch_size": args.batch_size,
        "device": device_str,
        "num_runs": args.num_runs,
        "use_half_precision": args.use_half,
        "avg_exec_time_sec": avg_time,
        "avg_gpu_memory_diff_MB": avg_gpu_mem,
        "avg_cpu_memory_diff_MB": avg_cpu_mem,
        "all_runs": {
            "times_sec": times,
            "gpu_mem_diff_MB": gpu_mem_diffs,
            "cpu_mem_diff_MB": cpu_mem_diffs
        }
    }

    # Generate a timestamped filename
    filename = f"metrics_{args.model}_{device_str}_{int(datetime.now().timestamp())}.json"
    out_path = os.path.join("results", filename)
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Benchmark complete.")
    print(f"Results saved to {out_path}")
    print(results)

if __name__ == "__main__":
    run_benchmark()
