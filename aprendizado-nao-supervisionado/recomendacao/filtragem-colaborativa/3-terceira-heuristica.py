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

#ordenando dataframe por nota_media e total_de_votos, ambos do maior para o menor
filmes = filmes.sort_values("nota_media",
    ascending=False).sort_values("total_de_votos", ascending=False)

#recomendando para um usuario que nao sabemos nada a respeito
# recomendação melhor que as duas primeiras
print("recomendacao:")
print(filmes.head(10))