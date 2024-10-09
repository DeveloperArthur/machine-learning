import torch

tns = torch.randn(2,2,3)

print(tns)
print(tns.size())

tns = tns.view(12) #2 x 2 é 4 e x 3 = 12
#passar -1 o view ja entende que é pra achatar tudo

print(tns)

tns = tns.view(4, 3) #4x3 = 12
#podemos redimensionar para qualquer outra dimensão, só precisa bater

print(tns)

tns = tns.view(6, 2) 
print(tns)

tns = tns.view(tns.size(0), -1) # isso significa: quero manter a primeira dimensao, mas quero achatar o restos
print(tns)