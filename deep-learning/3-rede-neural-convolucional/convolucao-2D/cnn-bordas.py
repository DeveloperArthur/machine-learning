from skimage import io, color, transform, data
from scipy.signal import convolve
import matplotlib.pyplot as plt
import numpy as np

img = data.brick()
plt.imshow(img, cmap='Greys') #tijolos.png

def show(valores, title):
  plt.figure(figsize=(len(valores), len(valores) ))
  plt.imshow(valores, cmap='gray')
  for i, line in enumerate(valores):
    for j, col in enumerate(line):
      plt.text(j, i, '{:.0f}'.format(col), fontsize=16, color='red', ha='center', va='center')
  plt.title(title)
  plt.xticks([])
  plt.yticks([])
  plt.savefig(title+'.png', format='png', dpi=100, bbox_inches='tight')

#Definindo os kernels para idenficiar bordas verticais e horizontais

#coluna preta, cinza e branca
kernel_vertical = [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]]
show(kernel_vertical, 'Kernel Vertical') #Kernel Vertical.png

#linhas preta, cinza e branca
kernel_horizontal = [[-1, -1, -1],
                     [0, 0, 0],
                     [1, 1, 1]]
show(kernel_horizontal, 'Kernel Horinzontal') #Kernel Horizontal.png
#Como funciona o deslocamento do Kernel na imagem: deslocamento.png
#muito parecido com 1D explicado no cnn-acelerometro.py

'''
lembrando que na hora da convolução o kernel se inverte
então na realidade estamos buscando bordas que vão
do claro pro escuro da esquerda pra direita
e do claro pro escuro de cima para baixo
'''

#Convolução
mapa_de_caracteristicas = convolve(img, kernel_vertical, mode='valid')
plt.imshow(mapa_de_caracteristicas, cmap='Greys')
'''
mapa-de-ativacoes-verticais.png
só estamos vendo as bordas verticais da imagem, praticamente
não dá pra ver as bordas horizontais
'''

mapa_de_caracteristicas = convolve(img, kernel_horizontal, mode='valid')
plt.imshow(mapa_de_caracteristicas, cmap='Greys')
'''
mapa-de-ativacoes-horizontais.png
agora buscando somente as bordas horizontais, e as bordas verticais são 
pouco visiveis, imagens completamente diferentes entre sí
'''

plt.show()

'''
Tem uma nuance que nós podemos observar que o “Kernel” de borda eles 
vão detectar tanto bordas do claro para o escuro, quanto do escuro 
para o claro, só que ativando de formas diferentes, veja exemplo
usando nosso Kernel vertical na logo da Alura: mapa-alura.png

Procurando bordas verticais na logo da Alura, o que acontece aqui 
quando a vemos o mapa de ativação: nas bordas vão do claro para o 
escuro ele dá uma ativação alta branca, a ativação é mais positiva. 
Na borda que faz do escuro para o claro, ele dá uma ativação muito 
baixa, é uma ativação negativa.

Mas eu ainda consigo visualizar apenas as bordas, do claro para o
escuro, e do escuro para o claro, Com a diferença de que elas vão 
dar ativações em extremos diferentes ou muito baixo ou muito alto

Mas perceba que a rede cumpriu seu propósito, não vemos linhas horizontais...  

Por isso é comum ver visualizações que dão o valor absoluto da saída 
para vermos só as bordas independente da direção delas, independente 
se o gradiente do claro para o escuro ou vice versa: valor-absoluto.png

Se eu pego só o valor absoluto da ativação eu estou considerando o ponto 
onde a ativação foi mais forte, então aqui eu só estou pegando os lugares 
onde independente se foi do claro para escuro, do escuro para o claro, ele 
vai representar da mesma forma pela força da ativação. 
Nós vamos ver só as bordas efetivamente
'''