import torch
from torch import nn
from sklearn import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

wine = datasets.load_wine()
data = wine.data
target = wine.target

print(data.shape, target.shape) #(178, 13) (178,)
# temos 178 amostras, cada amostra tem 13 caracteristicas
# e temos 178 rótulos

# 13 caracteristicas e 3 classes
print(wine.feature_names, wine.target_names)

# criando Multi-Layer Perceptron
class WineClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super(WineClassifier, self).__init__()

        self.hidden  = nn.Linear(input_size, hidden_size) # camada intermediária
        self.relu    = nn.ReLU()                          # ativação não linear
        self.out     = nn.Linear(hidden_size, out_size)   # camada de saída
        self.softmax = nn.Softmax()                       # para transformar em uma distribuição de probabilidades

    def forward(self, X):
        feature = self.relu(self.hidden(X))
        output = self.softmax(self.out(feature))
        return output

input_size = data.shape[1] # entrada é a quantidade de features
# essa info está na 2 dimensão do data.shape: (178,13)
# ou seja, é a mesma coisa que atribuir 13 ao input
hidden_size = 32 # porque sim
out_size = len(wine.target_names) # saída é a quantidade de classes

net = WineClassifier(input_size, hidden_size, out_size).to(device) #cast na GPU
print(net)

# instânciando uma função de perda
criterion = nn.CrossEntropyLoss().to(device) #cast na GPU

Xtensor = torch.from_numpy(data).float().to(device) #cast na GPU
Ytensor = torch.from_numpy(target).to(device) #cast na GPU

print(Xtensor.dtype)
print(Ytensor.dtype)

# Forward
pred = net(Xtensor)
print(pred.shape) # torch.Size([178, 3])
# 178 amostras com 3 probabilidades
# ou seja, para cada amostra, há 13 caracteristicas
# e para cada amostra há 3 probabilidades

# agora que temos a predição e o rótulo, podemos
# comparar a predição da rede com a função de perda
loss = criterion(pred, Ytensor)
print(loss) # tensor(1.2200, grad_fn=<NllLossBackward0>)
# 1.2200 é a perda média entre todos os valores

# a média de todas as perdas pra todos os pares de predição/rótulo

# se pegarmos somente as primeiras 30 amostras
loss = criterion(pred[:30], Ytensor[:30])
print(loss) # tensor(1.5514, grad_fn=<NllLossBackward0>)
# 1.5514 é a média das perdas pra esses valores

# a loss é uma métrica do quão bem nosso modelo foi no teste