import gym
import numpy as np
import random
from animation import print_frames

env = gym.make('Taxi-v1')

def preenche_tabela_do_QLearning():
    #inicializando a tabela Q com zeros
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1 # taxa de aprendizagem
    gamma = 0.6 # fator de desconto
    epsilon = 0.1 # chance de escolha aleatoria

    epochs, penalties = 0,0

    for i in range(100001): # vai rodar 100000 vezes o problema
        state = env.reset() #inicializacao aleatoria do ambiente
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # escolhe acao aleatoriamente
            else:
                # dentro da tabela do Q-Learning, olhando para a coluna state
                # pegando a acao que contem o maior valor
                action = np.argmax(q_table[state]) 
            
            next_state, reward, done, info = env.step(action) #o nome é next_state mas 
            # o correto seria estado atual, uma vez que env.step foi executado

            next_max = np.max(q_table[next_state]) # melhor valor no proximo estado
            old_value = q_table[state, action] # pega o valor referente a acao executada e estado "anterior"
            
            #atualiza a tabela Q com a formula do Q-Learning
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state #muda valor para o proximo estado
            epochs += 1

    print("[NO TREINAMENTO] total de acoes realizadas: {}".format(epochs))
    print("[NO TREINAMENTO] total de penalidade recebidas: {}".format(penalties))

    return q_table

#resolvendo problema com o aprendizado adquirido
q_table_preenchida = preenche_tabela_do_QLearning()

env.reset()
print("state: ", env.s)
state = env.s
epochs, penalties = 0,0
frames = []

done = False

while not done:
    action = np.argmax(q_table_preenchida[state])
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    #guarda os movimentos do taxi
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action, 
        'reward': reward
    })

    epochs += 1

print_frames(frames) #printa os movimentos do taxi
print("[RESULTADO FINAL] total de acoes realizadas: {}".format(epochs))
print("[RESULTADO FINAL] total de penalidade recebidas: {}".format(penalties))