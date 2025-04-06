# benchmark/compare_runs.py
import os
import sys
import subprocess
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline vs. optimized (half-precision) runs")
    parser.add_argument("--model", type=str, default="vgg16",
                        choices=["mobilenet_v2", "resnet50", "vgg16", "bert"],
                        help="Model architecture to test")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for input")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of inference runs")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Baseline
    baseline_cmd = [
        sys.executable,
        os.path.join("benchmark", "benchmark_runner.py"),
        "--model", args.model,
        "--batch_size", str(args.batch_size),
        "--num_runs", str(args.num_runs)
    ]
    print("Running BASELINE:", " ".join(baseline_cmd))
    subprocess.run(baseline_cmd)

    # 2) Optimized (half precision)
    half_cmd = baseline_cmd + ["--use_half"]
    print("Running OPTIMIZED (fp16):", " ".join(half_cmd))
    subprocess.run(half_cmd)

    print("\nDone! Check the 'results/' folder for two JSON files.")
    print("One is baseline, the other is half-precision.\n")

if __name__ == "__main__":
    main()
