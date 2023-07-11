#Modelos de aprendizado não-supervisionado também treinam e preveem...

#O "machine learning", essa ideia de "aprendizado de máquina" é referente aos modelos
# pois os modelos treinam (fit) com os dados, e após o treino, são capazes de prever
# pois "aprenderam" com os dados... 

#Se você olhar algumas implementações nesse projeto verá que parece muito mais uma extração 
# e analise de dados, pois não há nenhum processo de "aprendizagem" por meio dos dados
# pois não estamos utilizando modelos...

# Mas implementações dos modelos não fogem muito dessas implementações mais simples
# que temos nesse projeto... Essas implementações mais simples, por mais que não utilizem 
# modelo de machine learning, ainda são consideradas machine learning
# só que são implementações menos eficientes...

import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Carregar os dados dos arquivos CSV
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Filtrar os dados relevantes para o sistema de recomendação
ratings_filtered_df = ratings_df[['userId', 'movieId', 'rating']]
movies_filtered_df = movies_df[['movieId', 'title']]

# Criar uma tabela pivot para obter uma matriz de usuário-filme
pivot_table = ratings_filtered_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Instanciar o modelo KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(pivot_table.values)

# Predizer filmes com base no usuário existente
# Supondo que você queira recomendar filmes para o usuário de ID 1
user_id = 1
user_ratings = pivot_table.loc[user_id, :].values.reshape(1, -1)

# Encontrar os filmes mais similares aos do usuário
distances, indices = knn.kneighbors(user_ratings, n_neighbors=5)  # Defina o número de filmes a serem recomendados

# Obter os IDs dos filmes recomendados
recommended_movie_ids = pivot_table.columns[indices.flatten()]

# Encontrar os títulos dos filmes recomendados
recommended_movies = movies_filtered_df[movies_filtered_df['movieId'].isin(recommended_movie_ids)]

# Exibir os filmes recomendados
print("Filmes recomendados para o usuário", user_id)
print(recommended_movies[['movieId', 'title']])
