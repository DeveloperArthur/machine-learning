import torch
import numpy as np

arr = np.random.rand(3,4) #3 linhas e 4 colunas
tns = torch.from_numpy(arr)

print(type(tns))
print(arr)
print(arr.dtype)
print(tns)
print(tns.dtype)

#converter tensor para numpy

arr = tns.data.numpy()
print(type(arr))