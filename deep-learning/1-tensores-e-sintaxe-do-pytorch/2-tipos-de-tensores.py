import torch

lista = [[1, 2, 3],
         [4, 5, 6]]

tns = torch.Tensor(lista)
print(tns.dtype)
print(tns)

tns = torch.FloatTensor(lista)
print(tns.dtype)
print(tns)

tns = torch.DoubleTensor(lista)
print(tns.dtype)
print(tns)

tns = torch.LongTensor(lista)
print(tns.dtype)
print(tns)