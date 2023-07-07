import pandas as pd

filmes = pd.read_csv("movies.csv")
filmes.columns = ["filmeId", "titulo", "generos"]
filmes = filmes.set_index("filmeId")

notas = pd.read_csv("ratings.csv")
notas.columns = ["usuarioId", "filmeId", "nota", "momento"]

print(notas["filmeId"].value_counts()) #[filmeId, quantidade de notas]
total_de_votos = notas["filmeId"].value_counts()

#adicionando total_de_votos no dataframe de filmes
filmes['total_de_votos'] = total_de_votos

#ordenando dataframe por total_de_votos, do maior para o menor
filmes = filmes.sort_values("total_de_votos", ascending=False)

#primeira heuristica de recomendação, baseada em quantidade de avaliações
# a recomendação são os 5 primeiros filmes, eu indicariamos para 
# um usuario que não conhecemos, simplesmente indicamos os mais populares
# definimos popular como sendo mais avaliado
print("recomendação:")
print(filmes.head())