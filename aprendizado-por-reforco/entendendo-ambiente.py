import gym

env = gym.make('Taxi-v1')

env.render()

#ambiente: azul é onde está o passageiro
# rosa é o local onde o passageiro precisa ficar
# amarelo é o agente, nesse caso o taxi

#acoes: 
# 0= mover p baixo 
# 1=mover p cima 
# 2=mover p direita 
# 3=mover p esquerda 
# 4=pegar o passageiro 
# 5=largar o passageiro

#punições e recompensas:
# -1 : Cada movimento feito pelo carro ou tentativa de bater no muro
# -10 : Pegar ou largar o passageiro no lugar errado
# +20 : Deixar o passageiro no lugar certo

print("total de acoes: {}".format(env.action_space))
print("total de estados: {}".format(env.observation_space))

#alterando o estado manualmente:
#env.encode(linha, coluna, ponto-partida, ponto-chegada)
#R=0, G=1, Y=2, B=3
state = env.encode(3, 1, 1, 0)
print("numero do estado: ", state)

env.s = state
env.render()

env.s = 369 #os estados vao de 0 ate 499 pois 
# cada célula pode ter diferentes configurações
# como a presença ou ausência do passageiro e do destino
# e o local do táxi. Isso resulta em uma variedade de estados possíveis.

print("numero do estado: ", env.s)
env.render()

#mostra as probabilidades de transição
# o próximo estado, a recompensa recebida e 
# se o episódio termina após a transição para 
# cada ação possível no estado 369
print(env.P[env.s])