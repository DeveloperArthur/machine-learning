#Pegar uma distribuição dividida em duas classes e dizer qual é a reta que melhor divide esses dados
#Serão duas classes linearmente separáveis, para podermos treinar um modelo para classificar elas
#O exemplo está no arquivo 2-classificacao-linear.py

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(30)

X, Y = make_classification(n_features=2, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1)

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

p = X[10] # ponto 10 é amarelo
print(Y[10]) # 1 é amarelo

#fazendo uma reta bem proxima do correto (isso é um modelo, w1=0, w2=5, b=1, esse é o modelo)
w1 = 0
w2 = 5
b = 1
plotmodel(w1, w2, b)

p = (1, -2)
print(w1 * p[0] + w2 * p[1] + b)
#plt.plot(p[0], p[1], marker='^', markersize=20)
#o resultado desse ponto deu -9, é negativo
# e esta do lado azul, entao estamos fazendo uma boa classificação...

def classify(ponto, w1, w2, b):
    ret = w1 * ponto[0] + w2 * ponto[1] + b

    #negativo é azul
    if ret >= 0:
        return 1, 'yellow'
    else:
        return 0, 'blue'

#mais um ponto negativo

p = (2, 0)
classe, cor = classify(p, w1, w2, b)
print(classe, cor)

plotmodel(w1, w2, b)
#plt.plot(p[0], p[1], marker='^', markersize=20)
#Nossa classificação funcionou, classificou como 1 yello e está do lado amarelo..

acertos = 0
for k in range(len(X)):
    categ, _ = classify(X[k], w1, w2, b)
    if categ == Y[k]:
        acertos+=1

print("Acurácia: {0}%".format(100*acertos/len(X))) # deu acurácia de 92% (resultado-exercicio.png)

plt.show()

#Porque quando eu mexo no w1, w2, b a acurácia muda?
#Porque a classificação funciona com base na reta que eu criei (usando w1, w2, b)
#O ponto tal está para cima ou para baixo?
#Em outras palavras: O ponto 10 na minha reta foi classificado como amarelo,
# Eu vejo no Y qual é a cor do ponto 10 de verdade, se for amarelo eu acertei


