'''
Assuma um problema de classificação binária, onde é necessário classificar
modelos de carro entre as duas categorias:
- econômico, não econômico
Para isso, você possui os seguintes atributos de cada carro:
- peso, cilindrada, ano, potência
Sabendo disso, marque a seguir a alternativa que implementa o perceptron
adequado para representar uma potencial solução para o problema proposto.
'''

from torch import nn
perceptron = nn.Linear(4, 1)

#resposta correta!