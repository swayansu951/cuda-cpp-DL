import torch
import numpy as np
import os

ckpt = torch.load('ckpt.pt', map_location='cpu')
sd = ckpt['model']
os.makedirs('weights', exist_ok=True)

def dump(name, tensor, transpose=False):
    t = tensor.detach().float()
    if transpose:
        t = t.t().contiguous()
    arr = t.numpy().astype(np.float32)
    arr.tofile(f'weights/{name}.bin')
    print(f'{name}: shape {tuple(arr.shape)}, {arr.nbytes} bytes')

dump('wte', sd['transformer.wte.weight'])          # [65, 384] — embedding lookup, NOT a matmul, no transpose
dump('wpe', sd['transformer.wpe.weight'])          # [256, 384] — same, lookup only

for i in range(6):
    p = f'transformer.h.{i}.'
    dump(f'h{i}_ln1_gamma', sd[p+'ln_1.weight'])
    dump(f'h{i}_attn_qkv',  sd[p+'attn.c_attn.weight'], transpose=True)   # -> [384, 1152]
    dump(f'h{i}_attn_proj', sd[p+'attn.c_proj.weight'], transpose=True)  # -> [384, 384]
    dump(f'h{i}_ln2_gamma', sd[p+'ln_2.weight'])
    dump(f'h{i}_mlp_fc',    sd[p+'mlp.c_fc.weight'],   transpose=True)   # -> [384, 1536]
    dump(f'h{i}_mlp_proj',  sd[p+'mlp.c_proj.weight'], transpose=True)   # -> [1536, 384]

dump('lnf_gamma', sd['transformer.ln_f.weight'])
dump('lm_head', sd['lm_head.weight'], transpose=True)  # -> [384, 65]

# confirm tying before you assume it in C++ — don't just trust nanoGPT's default silently
print('wte and lm_head share memory:', sd['transformer.wte.weight'].data_ptr() == sd['lm_head.weight'].data_ptr())