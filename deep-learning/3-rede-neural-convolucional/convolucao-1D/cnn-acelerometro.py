'''
Você decidiu coletar dados do acelerômetro de um celular.

O objetivo é fazer com que pessoas caminhem com o celular no bolso para
analisar como o sensor responde a esse movimento.

A magnitude no sinal do acelerômetro se altera como uma espécie de senóide ruidosa.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve

x = np.linspace(0, 100, 100) #espaço de 0 a 100
y = 10 * np.sin(x) * np.random.rand(x.shape[0]) # com valores aleatorios

#senoide-ruidosa.png
plt.figure(figsize=(12, 4))
plt.plot(x, y)

#vamos em cima dessa senoide criar um filtro convolucional
# que vai conseguir detectar os intervalos crescentes desse sinal

def show(valores, title,):
    plt.figure(figsize=(len(valores), 2))
    plt.imshow(valores[np.newaxis, :], cmap='gray')
    for k, s in enumerate(valores):
        plt.text(k, 0, '{:.1f}'.format(s), fontsize=16, color='red', ha='center', va='center')
    plt.title(title, fontsize=18)
    plt.yticks([])

sinal = y[5:15] #pegando um pequeno trecho da senoite, de 5 á 15
show(sinal, 'Sinal') # #sinal.png cada ponto na imagem de cima
# equivale á um ponto no sinal embaixo, cinza escuro é baixo,
# cinza claro é alto, preto é o ponto mínimo e branco é o ponto máximo
# a intensidade da cor se refere á valores
plt.figure(figsize=(14,5))
plt.plot(sinal)

# o que vamos fazer é identificar os intervalos crescentes
# em pedaços do sinal e depois no sinal completo

'''
Kernel (filtro convolucional)
se a gente quer encontrar intervalos crescentes no nosso sinal
a gente tem que produzir um kernel que também carregue
esse padrão com ele, que também seja um intervalo crescente
justamente porque a operação da convolução vai medir a semelhança
entre dois sinais, então meu filtro tem que ter o sinal que
eu quero encontrar no meu dado
'''

kernel = np.asarray([1,0,-1])
show(kernel, 'Kernel') # kernel-decrescente.png
'''
esse kernel vai simular um intervalo decrescente
porque o que ele vai operar efetivamente
quando chamarmos a função convolução
é esse kernel invertido
'''

'''
Deslocando o kernel ao longo do sinal
só para entendermos passo a passo o que
a função convolução vai fazer:

essa operação basicamente é um produto 
de ponto a ponto, depois soma e tem o 
resultado, veja exemplos:
deslocamento.png
deslocamento-2.png
deslocamento-3.png

com isso estamos criando um sinal resultante 
que vai ser a nossa ativação
'''

#Agora vamos gerar automaticamente
#convoluir o sinal (linha 32) de entrada com o kernel
ativacao = convolve(sinal, kernel, mode='valid')
show(ativacao, 'Mapa de Ativação') #mapa-ativacao.png

#Pegamos exatamente o mapa de ativação e plotamos como imagem
#sobre o nosso sinal
plt.figure(figsize=(12, 4))
plt.plot(sinal, color='k', linewidth=4)
plt.imshow(ativacao[np.newaxis, :], cmap='Reds', aspect='auto',
           alpha=0.8, extent=(0.5, 8.5, -10, 10))
'''
mapa-com-sinal.png

essa imagem significa que onde tem imagem crescente
é onde ele vai ter suas ativações mais fortes
e a ativação mais fraca que ele vai ter é justamente
no ponto onde tem um sinal oposto ao sinal que o kernel
representa, se o kernel representa um intervalo crescente
o seu ponto mínimo vai ser no intervalo decrescente, que é
o oposto do que o kernel representa
'''

#Rodando convolve para uma parte maior do sinal:
ativacao = convolve(y[:50], kernel, mode='valid')
ativacao[ativacao < 0] = 0 # se todos os pontos da ativação forem < 0, deixe 0, pois só quero ativações positivas
plt.figure(figsize=(15, 4))
plt.plot(y[:50], color='k', linewidth=4)
plt.imshow(ativacao[np.newaxis, :], cmap='Reds', aspect='auto',
           alpha=0.8, extent=(0.5, 48.5, -10, 10))
plt.xlim(-0.5, 50.5)
plt.colorbar()
'''
sinal-completo.png

Nós temos nessa imagem uma espécie de classificação
dos intervalos crescentes do nosso sinal
uma imagem onde é forte onde tem intervalos crescentes
e é negativo onde não tem esses intervalos crescentes

Essa é uma simples convolução 1D para detectar 
intervalos crescentes em um sinal de acelerômetro
'''

plt.show()