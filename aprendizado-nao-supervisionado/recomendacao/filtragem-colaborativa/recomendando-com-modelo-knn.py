#Modelos de aprendizado não-supervisionado também treinam e preveem...

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
