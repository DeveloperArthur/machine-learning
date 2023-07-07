from sklearn.naive_bayes import MultinomialNB
import pandas as pd

data_frame = pd.read_csv('caes_e_porcos.csv')
X_df = data_frame[['gordinho','perninha_curta','faz_auau']]
Y_df = data_frame['eh_cao']

dados_de_treino = X_df[:5]
dados_de_treino_marcacao = Y_df[:5]

dados_de_teste = X_df[5:]
dados_de_teste_marcacao = Y_df[5:]

def calcula_taxa_de_acerto(resultado, dados_de_teste, dados_de_teste_marcacao):
    diferencas = resultado - dados_de_teste_marcacao
    acertos = 0
    for d in diferencas:
        if d == 0:
            acertos=acertos+1

    total_de_acertos = acertos
    total_de_elementos = len(dados_de_teste)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos
    return taxa_de_acerto

print("dados_de_treino:")
print(dados_de_treino)
print("dados_de_teste:")
print(dados_de_teste)

modelo = MultinomialNB()
modelo.fit(dados_de_treino, dados_de_treino_marcacao)

resultado = modelo.predict(dados_de_teste)
print("resultado da predição:")
print(resultado)
print("resposta correta:")
print(dados_de_teste_marcacao)

print("porcentagem de acerto:")
print(calcula_taxa_de_acerto(resultado, dados_de_teste, dados_de_teste_marcacao))