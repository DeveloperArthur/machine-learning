import torch
from torch import nn
from sklearn import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

diabetes = datasets.load_diabetes()
data = diabetes.data
target = diabetes.target

print(data.shape, target.shape)

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

input_size = data.shape[1]
hidden_size = 32 # porque sim
out_size = 1 # porque a gente vai fazer a regressão da progressão da diabetes

net = WineClassifier(input_size, hidden_size, out_size).to(device) #cast na GPU

criterion = nn.MSELoss().to(device)

Xtensor = torch.from_numpy(data).float().to(device)
Ytensor = torch.from_numpy(target).to(device)

# Forward
pred = net(Xtensor)
print(pred.shape)

# squeeze porque tanto rótulo quanto dado
# precisam ter a mesma dimensionalidade
loss = criterion(pred.squeeze(), Ytensor)
print(loss.data)

# 1-funcao-de-perda-classificacao.py tem comentários explicando linha-a-linha