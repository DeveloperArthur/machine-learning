import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

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

def distancia_de_usuarios(usuario_id1, usuario_id2, minimo_de_filmes_em_comum = 5):
    notas1 = notas_do_usuario(usuario_id1)
    notas2 = notas_do_usuario(usuario_id2)
    diferencas = notas1.join(notas2, lsuffix="_esquerda", rsuffix="_direita").dropna()

    if(len(diferencas) < minimo_de_filmes_em_comum):
        return None

    distancia = distancia_de_vetores(diferencas['nota_esquerda'], diferencas['nota_direita'])
    return [usuario_id1, usuario_id2, distancia]

quantidade_de_usuarios = len(notas["usuarioId"].unique())
print("quantidade de usuarios no dataset:")
print(quantidade_de_usuarios)

def distancia_de_todos_para(usuario_para_recomendar, n = None):
    distancias = []
    todos_os_usuarios = notas["usuarioId"].unique()
    
    #só vai usar os N primeiros elementos
    if n:
        todos_os_usuarios = todos_os_usuarios[:n]

    for usuario_id in todos_os_usuarios:
        distancias.append(distancia_de_usuarios(usuario_para_recomendar, usuario_id, 5))
    #retirando todos os usuarios None
    distancias = list(filter(None, distancias))
    distancias = pd.DataFrame(distancias, columns=["usuario_para_recomendar", "outro_usuario", "distancia"])
    return distancias

#N nao significa que vai retornar os N mais proximos
# vai retornar dos N primeiros usuarios, os que tem alguma 
# coisa parecida com o usuario_para_recomendar ordenado pela distancia
def mais_proximo_do(usuario_para_recomendar, n = None):
    distancias = distancia_de_todos_para(usuario_para_recomendar, n)
    distancias = distancias.set_index("outro_usuario").drop(usuario_para_recomendar)
    return distancias.sort_values("distancia")

#quando coloco n=5 ele trás 3 pq estamos excluindo os None
#print(mais_proximo_do(1, n=5))

def recomenda_para(usuario_para_recomendar, n = None):
    notas_do_usuario_a_recomendar = notas_do_usuario(usuario_para_recomendar)
    filmes_que_usuario_a_recomendar_ja_viu = notas_do_usuario_a_recomendar.index

    similares = mais_proximo_do(usuario_para_recomendar, n)
    id_usuario_mais_similar = similares.iloc[0].name
    print("id do usuario mais similar")
    print(id_usuario_mais_similar)
    notas_do_similar = notas_do_usuario(id_usuario_mais_similar)

    plt.plot(notas_do_usuario_a_recomendar, "go")
    plt.plot(notas_do_similar, "yo")
    plt.legend(["usuario_a_recomendar", "usuario_mais_similar"])
    plt.title("Distancia entre usuarios")

    #notas do similar retirando os filmes que o usuario a recomendar já viu
    notas_do_similar = notas_do_similar.drop(filmes_que_usuario_a_recomendar_ja_viu, errors='ignore')
    recomendacoes = notas_do_similar.sort_values("nota", ascending=False)
    return recomendacoes.join(filmes)

print(recomenda_para(30).head(10))
plt.show()