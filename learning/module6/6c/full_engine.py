import torch.nn.functional as F

x = sd['transformer.wte.weight'][[5,10,20,30]] + sd['transformer.wpe.weight'][:4]
ln = F.layer_norm(x, (384,), sd['transformer.h.0.ln_1.weight'], None, 1e-5)
qkv = ln @ sd['transformer.h.0.attn.c_attn.weight'].t()
print(qkv[0, :3])