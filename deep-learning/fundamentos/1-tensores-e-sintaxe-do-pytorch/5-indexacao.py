import torch

tns = torch.randn(3,3) #3 linhas e 3 colunas

print(tns)
tns[0, 2] = -10 #acessando linha 0 coluna 2 e trocando valor para -10
print('\n', tns)

print('\n', tns[0:2]) #acessando fatia do tensor, acessando as linhas 0 e 1 do tensor

print(tns[0,2]) # <- printando o valor presente na linha 0 coluna 2 (printando um tensor com dimensão 0)