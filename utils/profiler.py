# utils/profiler.py
import torch
import time
import pynvml
import psutil

def get_gpu_memory():
    print("⚠️ Skipping GPU memory check - running on CPU.")
    return 0  # Return dummy value since no GPU is present


def profile_model_run(model, input_tensor):
    start_gpu_mem = get_gpu_memory()
    start_cpu_mem = psutil.virtual_memory().used
    start_time = time.time()

    with torch.no_grad():
        _ = model(input_tensor)

    end_time = time.time()
    end_gpu_mem = get_gpu_memory()
    end_cpu_mem = psutil.virtual_memory().used

    metrics = {
        "exec_time_sec": end_time - start_time,
        "gpu_memory_diff_MB": end_gpu_mem - start_gpu_mem,
        "cpu_memory_diff_MB": (end_cpu_mem - start_cpu_mem) / 1024**2
    }
    return metrics
