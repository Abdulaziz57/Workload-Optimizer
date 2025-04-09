# benchmark/multi_benchmark_runner.py
import os
import sys
import subprocess
from datetime import datetime

"""
This script calls 'benchmark_runner.py' multiple times with different arguments,
so you can get multiple JSON results easily without manually typing commands.

Customize the 'configs' list below with the combinations you want to test.
"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # Tweak the combos to avoid super-long runs
    models_list = [
        "mobilenet_v2",
        "resnet50",
        # "vgg16",        # comment out or keep one big model
        # "efficientnet_b0",
        # "inception_v3",
    ]

    batch_sizes = [1, 2, 4]  # smaller set
    half_options = [False, True]
    num_runs = 3

    for model in models_list:
        for bs in batch_sizes:
            for half in half_options:
                cmd = [
                    sys.executable,
                    os.path.join("benchmark", "benchmark_runner.py"),
                    "--model", model,
                    "--batch_size", str(bs),
                    "--num_runs", str(num_runs),
                ]
                if half:
                    cmd.append("--use_half")

                print(f"\n=== RUNNING: {cmd} ===")
                subprocess.run(cmd)

if __name__ == "__main__":
    main()