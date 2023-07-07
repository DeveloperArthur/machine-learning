#calculando distancia euclidiana entre usuarios similares

import matplotlib.pyplot as plt 
import numpy as np

#notas que usuarios deram para 2 filmes
joao = np.array([4, 4.5])
maria = np.array([5, 5])
joaquina = np.array([3.5, 4.5])

def distancia(a,b):
    return np.linalg.norm(a - b)

print("distancia entre usuarios:")
#calculo de algebra linear
print(distancia(joao, maria))
print(distancia(joao, joaquina))

#joaquina é mais parecida com joao, portanto, na hora de 
# recomendar filmes para o joao, vou olhar os filmes da joaquina
# e baseado no que ela gostou, filtro os filmes, para recomendar algo para o joao

plt.plot(4, 4.5, "go")
plt.plot(5, 5, "yo")
plt.plot(3.5, 4.5, "bo")
plt.legend(["Joao", "Maria", "Joaquina"])
plt.title("Calcular a distancia entre usuarios")

plt.show()