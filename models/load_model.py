# models/load_model.py
import torch
import torchvision.models as models

def load_model_and_input():
    model = models.mobilenet_v2(pretrained=True).eval().to("cpu")
    dummy_input = torch.randn(1, 3, 224, 224).to("cpu")
    return model, dummy_input
