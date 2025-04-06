import torch
from torchvision import models

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model_and_input():
    device = get_device()
    model = models.mobilenet_v2(weights="IMAGENET1K_V1").eval().to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    return model, dummy_input
