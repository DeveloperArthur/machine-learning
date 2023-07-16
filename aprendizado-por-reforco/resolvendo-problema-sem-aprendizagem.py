import gym
import random
from animation import print_frames

env = gym.make('Taxi-v1')

env.s = 329

epochs = 0
penalties = 0
frames = []

done = False

while not done:
    action = random.randint(0, 5)  #escolhe aleatoriamente uma acao
    #action = env.action_space.sample() esse tbm dá, mas sempre vem os mesmos numeros, o de cima é aleatorio de verdade
    
    state, reward, done, info = env.step(action) #aplica a acao

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
print("total de acoes realizadas: {}".format(epochs))
print("total de penalidade recebidas: {}".format(penalties))

#nao estamos fazendo nada com as recompensas, ele esta 100% aleatorio