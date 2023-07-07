#exemplo bem simples de filtragem baseada em conteudo mas não é uma implementação eficiente

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

#id dos filmes que eu assisti
eu_assisti = [1, 33615, 26776, 1274, 1580, 5349, 8974]
ultimo_genero_que_eu_assisti = "Adventure|Animation|Children|Comedy"
print(filmes.loc[eu_assisti])

filmes_com_mais_de_50_avaliacoes = filmes.query("total_de_votos >= 50")
aventura_animacao_infantil_e_comedia = filmes_com_mais_de_50_avaliacoes.query("generos==%d" % ultimo_genero_que_eu_assisti)
recomendacao = aventura_animacao_infantil_e_comedia.drop(eu_assisti, errors='ignore').sort_values("nota_media", ascending=False)

#recomendacao baseada no genero do ultimo filme que foi assistido por mim
print("recomendacao:")
print(recomendacao.head(10))