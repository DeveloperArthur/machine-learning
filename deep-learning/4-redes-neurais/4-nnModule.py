from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
from torch import nn

X1, Y1 = make_moons(n_samples=300, noise=0.2)
plt.scatter(X1[:, 0], X1[:, 1], marker='o',
            c=Y1, s=25, edgecolors='k') #(make-moons.png)

class MinhaRede(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MinhaRede, self).__init__()

        #definir arquitetura
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        # gerar uma saida a partir do X
        hidden = self.relu(self.hidden(X))
        output = self.output(hidden)
        return output

input_size = 2
hidden_size = 16
output_size = 1

net = MinhaRede(input_size, hidden_size, output_size)
print(net)

#Forward
print(X1.shape) #(300, 2)
tensor_de_entrada = torch.from_numpy(X1).float()
pred = net(tensor_de_entrada)
print(pred.size()) # torch.Size([300, 1])

'''
Subindo informações na GPU

Para conseguir executar modelos maiores em tempo hábil, é preciso carregar 
as informações na GPU para que o processamento seja realizado por ela.
'''

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)

input_size = 2
hidden_size = 16
output_size = 1
net = MinhaRede(input_size, hidden_size, output_size)
net = net.to(device) # colocando a rede na GPU
print(net)

#Forward na GPU
tensor_de_entrada = torch.from_numpy(X1).float()
tensor_de_entrada = tensor_de_entrada.to(device) # colocando a entrada na GPU
pred = net(tensor_de_entrada)
print(pred.size()) # torch.Size([300, 1])

plt.show()