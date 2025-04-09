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
    # We'll define a list of models
    models_list = [
        "mobilenet_v2",
        "resnet50",
        "vgg16",
        "efficientnet_b0",
        "inception_v3",
        "bert",      # if you want to add NLP
        "gpt2",      # if you want to add GPT2
    ]

    # We'll define some batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    # We'll define how many runs you want for each config
    num_runs = 5

    # We might want to test with and without half precision
    half_options = [False, True]

    # Now we systematically generate commands
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
