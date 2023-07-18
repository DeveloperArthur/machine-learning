# Exemplo de ajuste de parâmetros usando o gradiente descendente

# Dados de treinamento (x) e suas marcações corretas (y)
train_data = [(1, 2), (2, 3), (3, 4), (4, 5)]
# Parâmetro a ser ajustado
parameter = 0

# Hiperparâmetros do algoritmo de otimização
learning_rate = 0.01
num_epochs = 100

# Loop de treinamento
for epoch in range(num_epochs):
    # Para cada dado de treinamento
    for x, y in train_data:
        # Faz a previsão usando o parâmetro atual
        prediction = parameter * x
        # Calcula o erro (diferença entre a previsão e a marcação correta)
        error = prediction - y
        # Atualiza o parâmetro usando o gradiente descendente
        parameter = parameter - learning_rate * error * x

# Após o treinamento, o valor do parâmetro foi ajustado

# Fazer previsões com o modelo treinado
test_data = [5, 6, 7]  # Dados de teste
predictions = []

for x_test in test_data:
    # Fazer a previsão usando o parâmetro ajustado
    prediction = parameter * x_test
    predictions.append(prediction)

print("Previsões:", predictions)
#resultado deve ser [6, 7, 8]