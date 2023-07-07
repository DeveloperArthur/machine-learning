import math

# [é gordinho?, tem perninnha curta?, faz auau?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 0, 1]
cachorro3 = [0, 0, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

# 1 = porco & -1 = cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

dado_para_classificar = [1, 0, 0] # porco

def calcula_distancia_euclidiana(a,b):
    #Calcula a distância euclidiana entre dois pontos
    distance = 0
    for i in range(len(a)):
        distance += math.pow((a[i] - b[i]), 2)
    return math.sqrt(distance)

def predict(dados, marcacoes, dado_para_classificar, k=5):
    #Realiza a classificação de um exemplo com base nos dados de treinamento
    distancias = []
    for i in range(len(dados)):
        distancia = calcula_distancia_euclidiana(dados[i], dado_para_classificar)
        distancias.append((distancia, marcacoes[i]))
    
    # Ordena as distâncias em ordem crescente
    distancias.sort(key=lambda x: x[0])  
    
    # Seleciona os k vizinhos mais próximos
    k_nearest = distancias[:k]  
    
    #O loop itera sobre os vizinhos mais próximos, representados por k_nearest. 
    # Para cada vizinho, obtemos sua classe, que está armazenada em neighbor[1]. 
    # Em seguida, usamos essa classe como chave no dicionário votes.
    votes = {}
    for neighbor in k_nearest:
        label = neighbor[1]
        print("classe do vizinho mais proximo")
        print(label)
        votes[label] = votes.get(label, 0) + 1
    
    #Retorna a classe com mais votos
    predicted_label = max(votes, key=votes.get)  
    
    return predicted_label

result = predict(dados, marcacoes, dado_para_classificar)

print(result) # deve printar 1

#esse algoritmo não é eficiente, as vezes não acerta 
# o resultado talvez a forma como estamos medindo a 
# distancia nao seja o ideal mas é um exemplo super 
# simples de como os modelos de classificacao 
# funcionam por baixo dos panos