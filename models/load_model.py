# models/load_model.py

import torch
from torchvision import models

# For an example small BERT model
try:
    from transformers import BertModel, GPT2Model
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_model_and_input(model_name="mobilenet_v2", batch_size=1, use_half=False):
    device_str = get_device()
    device = torch.device(device_str)

    # 1. Select model
    if model_name == "mobilenet_v2":
        net = models.mobilenet_v2(weights="IMAGENET1K_V1")
    elif model_name == "resnet50":
        net = models.resnet50(weights="IMAGENET1K_V1")
    elif model_name == "vgg16":
        net = models.vgg16(weights="IMAGENET1K_V1")
    elif model_name == "efficientnet_b0":
        net = models.efficientnet_b0(weights="IMAGENET1K_V1")
    elif model_name == "inception_v3":
        net = models.inception_v3(weights="IMAGENET1K_V1")
    elif model_name == "bert":
        if not TRANSFORMERS_AVAILABLE:
            raise ValueError("transformers not installed. Please pip install transformers.")
        net = BertModel.from_pretrained("bert-base-uncased")
    elif model_name == "gpt2":
        if not TRANSFORMERS_AVAILABLE:
            raise ValueError("transformers not installed. Please pip install transformers.")
        net = GPT2Model.from_pretrained("gpt2")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    net.eval()

    # 2. Move model to device
    net.to(device)

    # 3. If user wants half precision and device is CUDA
    if use_half and device_str == "cuda":
        net.half()

    # 4. Prepare dummy input
    if model_name in ["mobilenet_v2", "resnet50", "vgg16", "efficientnet_b0", "inception_v3"]:
        # Typical image shape
        # inception_v3 expects 299x299 typically, but let's keep 224 for consistency
        dummy_input = torch.randn(batch_size, 3, 224, 224)
    elif model_name == "bert":
        dummy_input = torch.randint(0, 1000, (batch_size, 16))  # seq_len=16
    elif model_name == "gpt2":
        dummy_input = torch.randint(0, 1000, (batch_size, 16))  # simplistic
    else:
        dummy_input = torch.randn(batch_size, 3, 224, 224)

    # If using half precision on CUDA, input should also be half
    if use_half and device_str == "cuda" and dummy_input.dtype == torch.float32:
        dummy_input = dummy_input.half()

    dummy_input = dummy_input.to(device)

    return net, dummy_input, device_str
