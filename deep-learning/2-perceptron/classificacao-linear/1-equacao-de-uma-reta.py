import numpy as np
import matplotlib.pyplot as plt

a = -1
b = 4
c = 0.4

def plotline (a, b, c):
    x = np.linspace(-2, 4, 50)
    # formula ax + by + c = 0 
    # A, B são pesos (w), C é o viés (bias)
    y = (-a*x -c)/b 

    plt.axvline(0, -1, 1, color='k', linewidth=1)
    plt.axhline(0, -2, 4, color='k', linewidth=1)
    plt.plot(x,y)
    plt.grid(True)

ponto1 = (2, 0.4)
retorno1 = a * ponto1[0] + b * ponto1[1] + c
print("%.2f" % retorno1) # <- qualquer ponto sobre a reta vale zero

ponto2 = (1, 0.6)
retorno2 = a * ponto2[0] + b * ponto2[1] + c
print("%.2f" % retorno2) # <- qualquer ponto acima da reta tem valor positivo

ponto3 = (3, -0.4)
retorno3 = a * ponto3[0] + b * ponto3[1] + c
print("%.2f" % retorno3) # <- qualquer ponto abaixo da reta tem valor negativo

plotline(a, b, c) # <- printa a reta

# printa os pontos azul, vermelho e verde
plt.plot(ponto1[0], ponto1[1], color='b', marker='o')
plt.plot(ponto2[0], ponto2[1], color='r', marker='o')
plt.plot(ponto3[0], ponto3[1], color='g', marker='o')

#a reta é um classificador linear
#a reta é uma fronteira de decisão entre duas regiões no espaço
plt.show()