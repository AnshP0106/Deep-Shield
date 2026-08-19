import torch


def run_inference(model, input_tensor, device):
    """
    Runs inference using the loaded AI model.
    """

    input_tensor = input_tensor.to(device)

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    return output