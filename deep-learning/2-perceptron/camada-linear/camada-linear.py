# Perceptron é a unidade fundamental de redes neurais
# redes neurais mais simples usam a unidade de classificacao linear como parte fundamental

#otimizamos na mão, chutando e ajustando o modelo necessário pra fazer a classificação correta

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(42)

#mesmo modelo da imagem clássica do perceptron
#3 entradas e 1 saida igual na imagem (perceptron.png)
perceptron = nn.Linear(in_features=3, out_features=1)
print(perceptron)

for nome, tensor in perceptron.named_parameters():
    print(nome, tensor.data) #tem 1 pesos com 3 valores e 1 viés com 1 valor, igual a imagem (perceptron.png)

print('')
print(perceptron.weight.data)
print(perceptron.bias.data)

#equação do modelo:
# w1 * x1 + w2 * x2 + w3 * x3 + b
def plot3d(perceptron):
    w1, w2, w3 = perceptron.weight.data.numpy()[0]
    b = perceptron.bias.data.numpy()

    X1 = np.linspace(-1, 1, 10)
    X2 = np.linspace(-1, 1, 10)

    X1, X2 = np.meshgrid(X1, X2)
    X3 = (b - w1*X1 - w2*X2) / w3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=180)

    ax.plot_surface(X1, X2, X3, cmap='plasma')

plot3d(perceptron) # esse é o perceptron na prática
#fizemos um plot 3d com os pesos do perceptron (plot-3d-com-pesos-do-perceptron.png)

#FORWARD: como usar e qual é a saída desse perceptron

X = torch.Tensor([0, -1, 2]) # coordenadas para plotar um ponto no espaço
y = perceptron(X) #forward (passou na camada) / Y é a saída

print(y) # vendo a classe do ponto (-0.2194) e ficou acima da camada (ponto-acima.png)

plot3d(perceptron)
plt.plot([X[0]], [X[1]], [X[2]], color='r', marker='^', markersize=20)

# se alterarmos o Tensor para 0,1,2 o ponto vai ficar positivo e embaixo da camada (ponto-abaixo.png)

plt.show()

'''
O perceptron automatiza todo processo que fizemos manualmente com classificação linear
A gente precisou:
- plotar um ponto aleatorio pra ver quais cores representam quais classes
- desenhar uma reta com w1, w2, e b
- pega um ponto aleatorio
- calcular ele na equacao da reta pra ver se sai positivo ou negativo
- faz a classificação pra todos os pontos da equação pra ver quanto nossa reta acerta
então o objetivo era achar a melhor reta...

O perceptron já faz tudo isso internamente, só passamos o ponto no espaço e ele retorna
positivo ou negativo
'''