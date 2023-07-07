import pandas as pd
import numpy as np

filmes = pd.read_csv("../movies.csv")
filmes.columns = ["filmeId", "titulo", "generos"]
filmes = filmes.set_index("filmeId")

notas = pd.read_csv("../ratings.csv")
notas.columns = ["usuarioId", "filmeId", "nota", "momento"]

def notas_do_usuario(usuario):
    notas_do_usuario = notas.query("usuarioId==%d" % usuario)
    notas_do_usuario = notas_do_usuario[["filmeId", "nota"]].set_index("filmeId")
    return notas_do_usuario

def distancia_de_vetores(a,b):
    return np.linalg.norm(a - b)

def distancia_de_usuarios(usuario_id1, usuario_id2):
    notas1 = notas_do_usuario(usuario_id1)
    notas2 = notas_do_usuario(usuario_id2)
    diferencas = notas1.join(notas2, lsuffix="_esquerda", rsuffix="_direita").dropna()
    return distancia_de_vetores(diferencas['nota_esquerda'], diferencas['nota_direita'])

distancia = distancia_de_usuarios(1, 4)

print("distancia entre os dois usuarios:")
print(distancia)