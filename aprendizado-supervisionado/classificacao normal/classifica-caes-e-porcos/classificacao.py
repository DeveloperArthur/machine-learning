# [é gordinho?, tem perninnha curta?, faz auau?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 0, 1]
cachorro3 = [0, 0, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 1 = cachorro & -1 = porco
marcacoes = [1, 1, 1, -1, -1, -1]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

#sera q esse eh cachorro ou porco?
misterioso1 = [1, 1, 1] #-1
misterioso2 = [1, 0, 0] #1
misterioso3 = [0, 0, 1] #-1

teste = [misterioso1, misterioso2, misterioso3]

marcacoes_teste = [-1, 1, 1]

resultado = modelo.predict(teste)

diferencas = resultado - marcacoes_teste
print(":>>>>>>>>>>>")
print(diferencas)

#acertos = [d for d in diferencas if d == 0]
acertos = 0
for d in diferencas:
    print('d:')
    print(d)
    if d == 0:
        acertos=acertos+1

total_de_acertos = acertos

total_de_elementos = len(teste)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(resultado)
print(diferencas)
print(taxa_de_acerto)