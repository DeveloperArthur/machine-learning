import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

filmes = pd.read_csv("../movies.csv")
notas = pd.read_csv("../ratings.csv")

vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(filmes[""])