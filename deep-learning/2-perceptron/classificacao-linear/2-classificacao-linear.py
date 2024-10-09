#Esse conceito é uma base para redes neurais, mas ele por si só não é uma rede neural
# Perceptron é a unidade fundamental de redes neurais
# redes neurais mais simples usam a unidade de classificacao linear como parte fundamental

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(46) # sem esse código,
# toda vez que rodamos o programa, ele vai gerar uma nova distribuição
# porque é aleatório mesmo, esse é o propósito da função
# mas com np.random.seed nós fixamos uma distribuição apenas (distribuicao-inicial.png)

X, Y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1)

print(X) #vários pares de coordenadas que vão definir pontos no espaço
print(Y) #1 ou 0 depende da classe de cada ponto


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


#identificando 0 e 1
p = X[10]
print(Y[10]) #vendo qual a classe do ponto 10
plt.plot(p[0], p[1], marker='^', markersize=20)
#pronto, identificamos que 0 é azul, e 1 é amarelo (classes-0-e-1.png)

#modelo para reta aleatória (reta-aleatoria.png) que não separa bem os dados
w1 = -3
w2 = 5
b = 3
plotmodel(w1, w2, b)

#precisamos ajustar os valores do modelo (w1, w2, b) pra ver como conseguimos separar bem os dados
w1 = 5
w2 = 1
b = -0.4
plotmodel(w1, w2, b)
#esse "chute" ficou assim (chute.png)

#como sabemos se nossos valores estão classificando bem:
#identificando qual lado possitivo e qual lado negativo da reta
p = (-1, 1) #ponto aleatório no espaço X=-1 Y=1

#pra ver o retorno desse ponto na equação da reta e ver se o retorno do ponto é negativo ou positivo
print(w1 * p[0] + w2 * p[1] + b) #p[0] é dimensão 1 e p[1] é dimensão 2 / resultado do print: -4.4

plt.plot(p[0], p[1], marker='^', markersize=20)

#o ponto -1, 1 na fórmula deu -4.4 (negativo), ou seja, deu negativo
#se formos ver a imagem que foi plotada (ponto-aleatorio-negativo.png) iremos concluir que
#valores negativos em relação á reta são da classe azul
#e valores positivos são da classe amarela
#com essas informações já podemos fazer a classificação

def classify(ponto, w1, w2, b):
    ret = w1 * ponto[0] + w2 * ponto[1] + b

    #sabemos que se o retorno foi >= 0, é amarelo, classe 1
    if ret >= 0:
        return 1, 'yellow'
    else:
        return 0, 'blue'

#plotando um ponto aleatório diferente
p = (2, -1)
classe, cor = classify(p, w1, w2, b)
print(classe, cor)

plotmodel(w1, w2, b)
plt.plot(p[0], p[1], marker='^', color=cor, markersize=30)

#E beleza, (teste-funcionou.png) vemos que classificou corretamente, ele é amarelo
# está do lado direito e a classificação funcionou...

#Vamos agora classificar para todos os pontos da distribuição
# nós vamos pegar nossa classificação e comparar com todos os pontos da distribuição
# pra ver se estamos classificando certo, e quantos estamos acertando
acertos = 0
for k in range(len(X)):
    categ, _ = classify(X[k], w1, w2, b)
    if categ == Y[k]:
        acertos+=1

print("Acurácia: {0}%".format(100*acertos/len(X))) #87%

plt.show()