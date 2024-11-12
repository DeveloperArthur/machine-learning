import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
from torchvision import datasets, transforms

VOC = datasets.VOCSegmentation('..',
                               year='2012',
                               image_set='val',
                               download=True,
                               transform=transforms.ToTensor(),
                               target_transform=transforms.ToTensor(), )

dado, rotulo = VOC[0]

print(VOC[0]) #isso printou um tensor gigante e no final o rótulo 7

dado, rotulo = VOC[0]
print(type(dado), type(rotulo)) # <class 'torch.Tensor'> <class 'torch.Tensor'>
print(dado.size(), rotulo.size()) # torch.Size([3, 366, 500]) torch.Size([1, 366, 500])

print(len(VOC)) #1449

dado = dado.permute(1,2,0)
#imagemaviao.png
plt.figure(figsize=(8,6))
plt.imshow(dado)
#segmentacao.png
plt.figure(figsize=(8,6))
plt.imshow(rotulo[0], cmap='gray')

plt.show()