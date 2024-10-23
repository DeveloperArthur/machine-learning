#usado quando precisamos colocar informacao na gpu
#pegar dados e jogar na gpu, que é o hardware acelerado
#que vai permitir um processamento mais rápido

import torch

tns = torch.randn(2,2,3)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

#assim estamos jogando informacao na gpu
#ele ira jogar a variavel na gpu
tns = tns.to(device)
print(tns)

#pra criar modelos robustos e maiores de redes neurais
#é extremamente importante o uso da gpu