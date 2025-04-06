# utils/profiler.py
import time
import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False

def get_gpu_memory(device_str):
    if device_str == "cuda" and NVML_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2
    else:
        return 0

def profile_model_run(model, input_tensor, device_str):
    start_gpu_mem = get_gpu_memory(device_str)
    start_cpu_mem = psutil.virtual_memory().used
    start_time = time.time()

    with torch.no_grad():
        _ = model(input_tensor)

    end_time = time.time()
    end_gpu_mem = get_gpu_memory(device_str)
    end_cpu_mem = psutil.virtual_memory().used

    metrics = {
        "exec_time_sec": end_time - start_time,
        "gpu_memory_diff_MB": end_gpu_mem - start_gpu_mem,
        "cpu_memory_diff_MB": (end_cpu_mem - start_cpu_mem) / 1024**2
    }

    return metrics
