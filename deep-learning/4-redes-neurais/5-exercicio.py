'''
Considere agora o seguinte conjunto de dados para classificação
de flores a partir de atributos estruturais:

(modelo-3d.png)

Implemente, usando implementação do nn.Sequential
uma arquitetura que suporte o treinamento para esse problema de classificação.
Sugiro uma arquitetura com uma camada linear escondida com seis neurônios,
uma ativação ReLU e uma camada linear de saída.
'''

#Resposta:
from torch import nn

input_size = 3  # porque se trata de um espaço com 3 dimensões
hidden_size = 8 # porque sim
output_size = 3 # porque como são 3 classes, vai ser 1 resposta por classe
# (quando era 2 classes, a resposta era 1 porque era binário)

net_sequential = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=output_size)
)