import torch

a = torch.empty(1, 3)
a[:, 1] = 2
a[:, 2] = 3 
print(a)
arr, idx = torch.sort(a, descending=True)
print(arr)
print(idx)