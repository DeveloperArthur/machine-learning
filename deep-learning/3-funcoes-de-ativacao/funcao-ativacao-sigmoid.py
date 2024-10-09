import torch
from torch import nn
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(46)

X, Y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1)

#método que estávamos utilizando para plotar a reta manualmente com classificação linear
def plotmodel(w1, w2, b):
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, edgecolors='k')

    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()

    x = np.linspace(-2, 4, 50)
    y = (-w1 * x - b) / w2

    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.plot(x, y)
    plt.grid(True)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

# como fazer isso no pytorch

perceptron = nn.Linear(2, 1)
sigmoid = nn.Sigmoid()

#Atribuindo os pesos para o perceptron
# w1 = 5
# w2 = 1
# b = -0.4
perceptron.weight = nn.Parameter(torch.Tensor([[5, 1]]))
perceptron.bias = nn.Parameter(torch.Tensor([-0.4]))

print(perceptron.weight.data)
print(perceptron.bias.data)

markers=['^', 'v', '>', '<']
colors=['r', 'g', 'b', 'gray']

plotmodel(perceptron.weight.data[0][0].numpy(),
          perceptron.weight.data[0][1].numpy(),
          perceptron.bias.data[0].numpy())

for k, idx in enumerate([17, 21, 43, 66]):
    x = torch.Tensor(X[idx])

    ret = perceptron(x)
    act = sigmoid(ret) #ativando a saída do perceptron

    act_limiar = 0 if ret.data < 0 else 1

    label = 'ret: {:5.2f}'.format(ret.data.numpy()[0]) + ' limiar: {:4.2f}'.format(act_limiar) + ' act: {:5.2f}'.format(act.data.numpy()[0])
    plt.plot(x[0], x[1], marker=markers[k], color=colors[k], markersize=10, label=label)

plt.legend()
plt.show()

'''
Na classificação linear há apenas duas classes 0 e 1
Ao utilizar funções de ativações, como sigmoide, a saída
pode ser interpretada como uma probabilidade, mas ainda
representa essas duas classes

Veja na imagem (sigmoid-vs-limiar.png), o ponto vermelho
está á -18.03 da reta, o limiar classificou como 0, e a 
AF (activation function) classificou como 0...

O ponto verde está á  -1.54 da reta, o limiar classificou
como 0, a AF classificou como 0.18 porque "não está tão
longe da reta assim", a classe é 0, mas o valor varia
pra sinalizar quão próximo da reta está, é isso que a
função de ativação faz

Quando aplicamos a função de ativação sigmoide, ela 
transforma o valor limiar em um valor de probabilidade 
entre 0 e 1
'''