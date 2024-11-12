import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
from torchvision import datasets, transforms

'''
Iremos utilizar o dataset MNIST (mnist.png) que possui 10 classes

O dado nesse caso é a imagem, vamos receber uma imagem de entrada
e o rótulo, o objetivo do treinamento seria 1 inteiro, que seria
a classe que aquele dígito pertence

Dado uma imagem, qual seria o inteiro correspondente
'''

MNIST = datasets.MNIST('..',
                       train=False,  #não quero conjunto de treino, quero conjunto de teste
                       transform=transforms.ToTensor(),
                       download=True) #baixando os dados

print(MNIST[0]) #isso printou um tensor gigante e no final o rótulo 7
# ou seja, esse conjunto de dados representa um 7 (print.png)

#carregando de forma mais organizada colocando em uma tupla
dado, rotulo = MNIST[0]
print(type(dado), type(rotulo)) # <class 'torch.Tensor'> <class 'int'>
print(dado.size(), rotulo) # torch.Size([1, 28, 28]) 7

print(len(MNIST))

#plotando os 10 primeiros elementos do dataset, ele tem comprimento 10.000 (conjunto de teste)
fig, axs = plt.subplots(1, 10, figsize=(15, 4))
for i in range(10):
    dado, rotulo = MNIST[i]
    axs[i].imshow(dado[0], cmap='gray')
    axs[i].set_title(str(rotulo))
    # numeros-rotulos.png

plt.show()

'''
Como deve ser a última camada de uma rede cujo objetivo é 
classificar os dados do MNIST?
Lembrando dos MLP's, a última camada deve ter o número
correspondente ao número de classes
vejam representacao-rede-classificacao.png
o que tem que vir no final com certeza é uma camada de MLP
totalmente conectada, com 10 neurônios, 1 pra cada
categoria do meu dataset, teremos o neurônio que vai ativar
se o digito for 0, se for 1, se for 2 e daí por diante
'''