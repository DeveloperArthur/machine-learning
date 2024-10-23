import torch

tns1 = torch.randn(3,3)
tns2 = torch.randn(3,3)

#ver dimensao com .shape
print(tns1.shape)
print(tns2.shape)

#operacoes matematicas com tensores
print(tns1 + tns2)

print(tns1 - tns2)

print(tns1 * tns2) # <- faz multiplicacao ponto a ponto

print(tns1 / tns2)

#para fazer multiplicacao de produto interno:
print(torch.mm(tns1, tns2))