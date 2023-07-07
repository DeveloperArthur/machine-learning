#uma implementação / variação do algoritmo KNN

import pandas as pd
import numpy as np

# Configura para exibir todas as linhas
pd.set_option('display.max_rows', None)
# Configura para exibir todas as colunas
pd.set_option('display.max_columns', None)

filmes = pd.read_csv("movies.csv")
filmes.columns = ["filmeId", "titulo", "generos"]
filmes = filmes.set_index("filmeId")

notas = pd.read_csv("ratings.csv")
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

#knn vai pegar os k mais proximos do usuario_para_recomendar
# N nao significa que vai retornar os N mais proximos
# vai retornar dos N primeiros usuarios, os que tem alguma 
# coisa parecida com o usuario_para_recomendar ordenado pela distancia
def knn(usuario_para_recomendar, k = 10, n = None):
    distancias = distancia_de_todos_para(usuario_para_recomendar, n)
    distancias = distancias.set_index("outro_usuario").drop(usuario_para_recomendar, errors='ignore')
    return distancias.sort_values("distancia").head(k)

#quando coloco n=5 ele trás 3 pq estamos excluindo os None
#print(mais_proximo_do(1, n=5))

def recomenda_para(usuario_para_recomendar, k = 10, n = None):
    notas_do_usuario_a_recomendar = notas_do_usuario(usuario_para_recomendar)
    filmes_que_usuario_a_recomendar_ja_viu = notas_do_usuario_a_recomendar.index

    similares = knn(usuario_para_recomendar, k, n)
    usuarios_similares = similares.index
    print("ids dos usuarios similares:")
    print(usuarios_similares)
    notas_dos_similares = notas.set_index("usuarioId").loc[usuarios_similares]
    
    #notas dos similares retirando os filmes que o usuario a recomendar já viu
    notas_dos_similares = notas_dos_similares.drop(filmes_que_usuario_a_recomendar_ja_viu, errors='ignore')
    media_das_notas_dos_similares = notas_dos_similares.groupby("filmeId").mean()[["nota"]]
    
    #quantidade de usuarios similares que assistiram
    aparicoes = notas_dos_similares.groupby("filmeId").count()[["nota"]]
    filtro_minimo = k / 2

    #junta a media com as aparicoes
    recomendacoes = media_das_notas_dos_similares.join(aparicoes, lsuffix="_media_dos_usuarios", rsuffix="_quant_de_usuarios_que_assistiram")
    
    #antes recomendava varios filmes com nota 5, mas só 1 usuario tinha assistido e votado
    # para evitar esse cenario estamos filtrando pelas aparicoes
    recomendacoes = recomendacoes.query("nota_quant_de_usuarios_que_assistiram >= %.2f" % filtro_minimo)

    recomendacoes = recomendacoes.sort_values("nota_media_dos_usuarios", ascending=False)
    #faz o join para mostrar o titulo dos filmes
    return recomendacoes.join(filmes)

#dos N primeiros, trás quem sao os K mais proximos
# caso traga um numero menor que K, é pq 
# o restante nao tem similaridade suficiente (minimo_de_filmes_em_comum)
recomendacao = recomenda_para(1, k = 10)
print("recomendacoes geradas:")
print(len(recomendacao))
print("recomendacoes:")
print(recomendacao.head(10))