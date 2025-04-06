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

# Make sure Python can find your modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    # We define a list of (model, batch_size, num_runs, use_half) combos
    configs = [
        ("mobilenet_v2", 1, 3, False),
        ("mobilenet_v2", 1, 3, True),
        ("resnet50",     1, 3, False),
        ("resnet50",     1, 3, True),
        ("vgg16",        2, 3, False),
        # Add more as needed
    ]

    for (model, bs, runs, half) in configs:
        cmd = [
            sys.executable,  # your python interpreter
            os.path.join("benchmark", "benchmark_runner.py"),
            "--model", model,
            "--batch_size", str(bs),
            "--num_runs", str(runs),
        ]
        if half:
            cmd.append("--use_half")

        print(f"\n=== RUNNING: {cmd} ===")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
