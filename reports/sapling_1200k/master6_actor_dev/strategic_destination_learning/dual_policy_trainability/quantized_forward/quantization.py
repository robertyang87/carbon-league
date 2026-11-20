"""The production int8 array transform in PyTorch, with a straight-through derivative."""
import torch


def fake_quantize(weight):
    if weight.ndim < 2 or weight.numel() < 4096:
        return weight
    # Keep the master weight's gradient, including at saturation. Scale is measured, not trained.
    with torch.no_grad():
        flat = weight.flatten(1)
        scale = flat.abs().amax(1, keepdim=True) / 127.
        scale = torch.where(scale == 0, 1., scale)
        quantized = ((flat / scale).round().clamp(-127,127) * scale).reshape_as(weight)
    return quantized + (weight - weight.detach())


def parameters_for(module):
    return {name: fake_quantize(parameter) for name,parameter in module.named_parameters()}
