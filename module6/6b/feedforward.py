import torch
# we have to compare the torch's inbuilt gelu witht our gelu from scratch..
# torch.nn.functional.gelu(input=None,approximate=None)

model = torch.load('ckpt.pt', map_location='cuda')

sd = model['model'] if 'model' in model else model

for name, tensor in sd.items():
    print(name, tuple(tensor.shape))

print("\n",model.get('model_args', 'no model_args key found'))