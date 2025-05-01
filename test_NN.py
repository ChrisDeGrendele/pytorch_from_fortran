import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()
print("Model loaded and set to evaluation mode.")

def run_resnet_inference(input_tensor):
    """Runs inference on the pre-loaded ResNet18 model.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).

    Returns:
        torch.Tensor: The raw output tensor from the model.
    """
    with torch.no_grad():
        output = model(input_tensor)
    return output

if __name__ == "__main__":
    # ResNet18 expects input of shape (batch_size, 3, 224, 224)
    dummy_input = torch.ones(1, 3, 224, 224)
    output = run_resnet_inference(dummy_input)

    print("Inference completed!")
    print("Output shape:", output.shape)
    print("Python Raw Output (first 10):", output[0, :10].numpy())
    print("Python Raw Output (last 10):", output[0, -10:].numpy())

    
