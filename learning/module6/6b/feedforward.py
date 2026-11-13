import torch
# we have to compare the torch's inbuilt gelu witht our gelu from scratch..
# torch.nn.functional.gelu(input=None,approximate=None)

model = torch.load('ckpt.pt', map_location='cuda')

sd = model['model'] if 'model' in model else model

#  run this for "6b/"
# for name, tensor in sd.items():
#     print(name, tuple(tensor.shape))

# print("\n",model.get('model_args', 'no model_args key found'))

#  Run this for "6c/"
import torch.nn.functional as F

x = sd['transformer.wte.weight'][[5,10,20,30]] + sd['transformer.wpe.weight'][:4]
ln = F.layer_norm(x, (384,), sd['transformer.h.0.ln_1.weight'], None, 1e-5)
qkv = ln @ sd['transformer.h.0.attn.c_attn.weight'].t()
print(qkv[0, :3])