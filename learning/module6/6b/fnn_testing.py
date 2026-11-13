import torch

sd = torch.load("ckpt.pt", map_location="cpu", weights_only=True)

print("Model keys:")
print(sd['model'].keys())

print((sd['model']['transformer.wte.weight'][5] + sd['model']['transformer.wpe.weight'][0])[:3])