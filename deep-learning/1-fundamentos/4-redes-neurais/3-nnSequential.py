from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
from torch import nn

X1, Y1 = make_moons(n_samples=300, noise=0.2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o',
            c=Y1, s=25, edgecolors='k') #(make-moons.png)

input_size = 2  # a entrada da minha rede são 2 valores, X1 e X2
hidden_size = 16 # a camada hidden terá 16 neurônios, entao a saida
# da primeira camada vai ser 16, e a entrada da última camada vai ser 16 (representacao-rede-neural.png)
output_size = 1 # a saída da rede será 1
# porque o problema que queremos resolver é classificação binária (make-moons.png)
# podemos treinar um único perceptron pra interpretar a saída tipo "quanto mais próximo de 1 é uma classe
# e quando mais próximo de 0 é outra classe"
# o perceptron vai ser a última camada e vai retornar 1 saída

# (representacao-rede-neural.png)
net = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size), # camada escondida (hidden)
                    nn.ReLU(), # ativação não linear
                    nn.Linear(in_features=hidden_size, out_features=output_size)) # output (saída)
print(net)

from torchsummary import summary
summary(net, input_size=(1, input_size))
'''
summary printou que a quantidade de parâmetros total da rede é 65

48 da camada escondida porque o cálculo é: 
entradas X neuronios na camada escondida = 2x16=32
e cada neuronio vai ter um Bias = 16
32+16=48
 
17 da camada de saída porque o cálculo é:
neuronios na camada escondida X neuronios na camada de saida = 16x1=16
e cada neuronio vai ter um Bias = 1
16+1=17

48+17=65
'''

#Forward
print(X1.shape) #(300, 2)
tensor_de_entrada = torch.from_numpy(X1).float() # lembrando que X1 é um array que armazena
# todas as coordenadas dos pontos do gráfico
pred = net(tensor_de_entrada) #passamos todos os pontos do gráfico pra rede
print(pred.size()) #saída: torch.Size([300, 1])
# porque X1 tem 300 amostras com 2 dimensões
# essas 300 amostras passaram na rede e se
# transformaram em 300 predições, a predição
# só tem 1 dimensão (uma classe)

# ou seja, pra 300 dados de 2 dimensões, eu tenho
# 300 predições de um único perceptron na camada de saída

# isso é apenas a primeira parte do processo, não resolvemos o problema
# de classificação ainda



plt.show()