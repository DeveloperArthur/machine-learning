from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch import optim

features = [0, 9]

wine = datasets.load_wine()
data = wine.data[:, features]
targets = wine.target

# tres-classes-de-vinhos.png
# plt.scatter(data[:, 0], data[:, 1], c=targets, s=15, cmap=plt.cm.brg)
# plt.xlabel(wine.feature_names[features[0]])
# plt.ylabel(wine.feature_names[features[1]])

###### ----- Normalização (pre-processamento é importante para normalização dos valores de entrada)

# Sem isso não conseguimos chegar na otimização certa, fica parecendo o aleatório (veja fronteira-de-decisao-aleatoria2.png)
scaler = StandardScaler()
data = scaler.fit_transform(data)

# dados-normalizados.png (com um intervalo numérico mais parecido)
# plt.scatter(data[:, 0], data[:, 1], c=targets, s=15, cmap=plt.cm.brg)
# plt.xlabel(wine.feature_names[features[0]])
# plt.ylabel(wine.feature_names[features[1]])

###### ----- Instanciando a rede

import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

# 1-funcao-de-perda-classificacao.py tem comentários explicando linha-a-linha
input_size = data.shape[1]
hidden_size = 32
out_size = len(wine.target_names)

# ./1-fundamentos/4-redes-neurais/3-nnSequential.py tem comentários explicando linha-a-linha
net = nn.Sequential(nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, out_size),
                    nn.Softmax())
print(net)
net = net.to(device)

###### ----- Visualizando a fronteira de decisão

#Função auxiliar para plot da fronteira de decisão do classificador.
def plot_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    spacing = min(x_max - x_min, y_max - y_min) / 100

    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    data = np.hstack((XX.ravel().reshape(-1, 1),
                      YY.ravel().reshape(-1, 1)))

    # For multi-class problems
    db_prob = model(torch.Tensor(data).to(device))
    clf = np.argmax(db_prob.cpu().data.numpy(), axis=-1)

    Z = clf.reshape(XX.shape)

    plt.contourf(XX, YY, Z, cmap=plt.cm.brg, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=25, cmap=plt.cm.brg)

plot_boundary(data, targets, net) # fronteira-de-decisao-aleatoria.png
# cada vez que chamamos esse método, uma nova fronteira é formada pq
# está jogando uma fronteira aleatoriamente

###### ----- Otimizando a fronteira de decisão pra separar os grupos de dados
# separar as 3 classes de vinho, fazendo otimização
# temos 3 classes de vinho distribuídas no espaço e queremos classificar elas

#Função de perda
criterion = nn.CrossEntropyLoss().to(device)

#Otimizador: Descida do Gradiente
#Stochastic Gradient Descent
optimizer = optim.SGD(net.parameters(), lr=1e-3)

# Cast dos dados
Xtensor = torch.FloatTensor(data).to(device)
Ytensor = torch.LongTensor(targets).to(device)

# A otimização de redes neurais é um processo iterativo
# são passos pequenos,em 1 iteração de treinamento atualizar
# os pesos apenas 1 vez só, é quase imperceptível, quase não
# muda nada no seu modelo, é apenas 1 passo de otimização...
# 100 iterações é suficiente nesse caso
for i in range(100):
    # Forward
    pred = net(Xtensor)
    loss = criterion(pred, Ytensor)

    #Backward
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        plt.figure()
        plot_boundary(data, targets, net)
        plt.show()

# rede-otimizada.png

'''
Lembrando que isso não é tudo, o processo de treinamento
de uma rede neural passa pelos conceitos de iteração, batch, época 
aqui nós só fizemos a iteração...

um fluxo de treinamento comum seria assim:
# Epochs
for i in range(num_epochs):
    # Iterations
    for batch in train_data:
        #Forward
        ypred = net(batch)
        loss = criterion(ypred, y)
        
        #Backpropagation
        loss.backward()
        optimizer.step()

- batch é o conjunto de amostras vistas em uma única iteração 
- uma época é completada quando todas as amostras de 
treino foram utilizadas em pelo menos uma iteração

"Ah mas porque só a iteração não é suficiente e é necessário ver as 
mesmas amostras múltiplas vezes?"

sim, para otimizar um modelo é necessário ver as mesmas 
amostras múltiplas vezes, ou seja, múltiplas épocas
como a otimização é um processo iterativo, dando pequenos 
passos na superfície de erro, ver as amostras uma única vez 
não é suficiente para alcançar um ponto mínimo de erro

dar pequenos passos é importante, pois a cada iteração o 
modelo conta apenas com a informação local daquele batch. 
Ao final de cada época o modelo está um pouco mais próximo 
da solução, sendo ideal otimizar durante múltiplas épocas.

e além disso, ainda tem o fluxo de validação e 
o cálculo de acurácia pra medir o desempenho do modelo  

'''