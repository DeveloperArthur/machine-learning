import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
from torchvision import datasets, transforms

VOC = datasets.VOCDetection('..',
                            image_set='train',
                            download=True,
                            transform=transforms.ToTensor(), )

print(VOC[0])

dado, rotulo = VOC[0]
print(type(dado), type(rotulo)) # <class 'torch.Tensor'> <class 'dict'>
print(dado.size()) # torch.Size([3, 442, 500])

print(len(VOC)) #5717

dado = dado.permute(1, 2, 0)
plt.figure(figsize=(8, 6))
plt.imshow(dado)
#imagemVOC.png

print(rotulo) #o rótulo é um dicionário de várias coisas
#cada imagem vai ter mais de um objeto, diferentes informaçõe
#no dicinário tem um campo chamado 'bndbox'
#esse é o campo que define a caixa delimitadora
#ele tem xmax, ymax, xmin, ymin

#carregando a caixa delimitadora (delimiter-.png)
#o objeto 0 representa o cavalo, entao a caixa
#correspondente é do caval
bbox = rotulo['annotation']['object'][0]['bndbox']
xmax = int(bbox['xmax'])
xmin = int(bbox['xmin'])
ymax = int(bbox['ymax'])
ymin = int(bbox['ymin'])

fig, axs = plt.subplots(figsize=(8, 6))
axs.imshow(dado)

w, h = xmax-xmin, ymax-ymin
rect = patches.Rectangle((xmin, ymin), w,h, fill=False, color='r', linewidth=4)
axs.add_patch(rect)

plt.show()

'''
Como deve ser a última camada da rede cujo objetivo é
detectar objetos do PASCAL VOC?
Dessa vez, invés de classificação, vamos resolver
um problema de detecção, então, precisamos descobrir
qual é o valor de xmin, xmax, ymin, ymax
ou seja, 4 neurônios no final, cada um referênte
á variável que eu quero fazer regressão, e cada
neurônio, a intensidade de ativação tem que
correponder ao valor dessa coordenada
veja representacao-rede-deteccao.png
a rede convolucional conecta nos neurônios
para fazer a regressão de 4 valores
'''