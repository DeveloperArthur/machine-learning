import pandas as pd

filmes = pd.read_csv("../movies.csv")
filmes.columns = ["filmeId", "titulo", "generos"]
filmes = filmes.set_index("filmeId")

notas = pd.read_csv("../ratings.csv")
notas.columns = ["usuarioId", "filmeId", "nota", "momento"]

print(notas["filmeId"].value_counts()) #[filmeId, quantidade de notas]
total_de_votos = notas["filmeId"].value_counts()

#media de todos os filmes
notas_medias = notas.groupby("filmeId").mean()["nota"]

#adicionando total_de_votos e nota_media no dataframe de filmes
filmes['total_de_votos'] = total_de_votos
filmes["nota_media"] = notas_medias

#ordenando dataframe por nota_media, do maior para o menor
filmes = filmes.sort_values("nota_media", ascending=False)

#recomendando para um usuario que nao sabemos nada a respeito
# recomendação ruim, pois estamos recomendando filmes com a maior nota
# porém, poucas pessoas assistiram...
print("recomendacao:")
print(filmes.head(10))

#segunda tentativa de recomendação: filmes com a maior nota fltrando total_de_votos
filmes_com_mais_de_50_avaliacoes = filmes.query("total_de_votos >= 50")
print("recomendacao:")
print(filmes_com_mais_de_50_avaliacoes.sort_values("nota_media", ascending=False).head(10))